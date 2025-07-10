"""
Converts a Video to SuperSloMo version

Example usage:
python3 cv2-converter.py path_to_input --checkpoint=path_to_model --output=path_to_output --scale=
"""
from time import time
import click
import cv2
import torch
from PIL import Image
import numpy as np
import model
from torchvision import transforms
from torch.nn.functional import sigmoid
import datetime

torch.serialization.add_safe_globals([datetime.datetime])
torch.set_grad_enabled(False)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

image_to_tensor = transforms.ToTensor()
tensor_to_image = transforms.ToPILImage()

if device != "cpu":
    normalization_mean = [0.429, 0.431, 0.397]
    denormalization_mean = [-m for m in normalization_mean]
    normalization_std = [1] * 3
    image_to_tensor = transforms.Compose([image_to_tensor, transforms.Normalize(mean=normalization_mean, std=normalization_std)])
    tensor_to_image = transforms.Compose([transforms.Normalize(mean=denormalization_mean, std=normalization_std), tensor_to_image])

# Neural network models
compensate_flow = model.UNet(6, 4).to(device) 
interpolate = model.UNet(20, 5).to(device)     
backward_warping : model.backWarp              # instantiated during convert_video

def calculate_average_fps(current_avg, count, new_value):
    return (current_avg * count/(count+1) + new_value / (count+1), count+1)

def load_models(checkpoint_path):
    """Load and quantize the neural network models from checkpoint."""
    global interpolate, compensate_flow
    
    model_states = torch.load(checkpoint_path, map_location='cpu')
    interpolate.load_state_dict(model_states['state_dictAT'])
    compensate_flow.load_state_dict(model_states['state_dictFC'])

    interpolate.eval()
    compensate_flow.eval()

def load_batch(video_capture, batch_size, current_batch, target_width, target_height):
    """Load a batch of frames from the video capture."""
    if len(current_batch) > 0:
        current_batch = [current_batch[-1]]  # Keep last frame for temporal continuity

    for frame_idx in range(batch_size):
        success, raw_frame = video_capture.read()
        if not success:
            break
        # Convert BGR to RGB and preprocess
        rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        resized_frame = pil_frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
        rgb_converted_frame = resized_frame.convert('RGB')
        tensor_frame = image_to_tensor(rgb_converted_frame)
        current_batch.append(tensor_frame)

    return current_batch

def interpolate_batch(frame_batch, interpolation_factor):
    """Generate intermediate frames between consecutive frames in the batch."""
    previous_frames = torch.stack(frame_batch[:-1]).to(device)
    next_frames = torch.stack(frame_batch[1:]).to(device)

    concatenated_frames = torch.cat([previous_frames, next_frames], dim=1)

    # Compute optical flow between consecutive frames
    optical_flow_output = compensate_flow(concatenated_frames)
    forward_flow = optical_flow_output[:, :2, :, :]  # Flow from frame0 to frame1
    backward_flow = optical_flow_output[:, 2:, :, :]  # Flow from frame1 to frame0

    interpolated_frames = []
    for intermediate_idx in range(1, interpolation_factor):
        time_ratio = intermediate_idx / interpolation_factor
        quadratic_term = -time_ratio * (1 - time_ratio)

        # Compute intermediate optical flows
        intermediate_forward_flow = quadratic_term * forward_flow + (time_ratio * time_ratio) * backward_flow
        intermediate_backward_flow = ((1 - time_ratio) * (1 - time_ratio)) * forward_flow + quadratic_term * backward_flow

        warped_previous_frame = backward_warping(previous_frames, intermediate_forward_flow)
        warped_next_frame = backward_warping(next_frames, intermediate_backward_flow)

        # Prepare input for interpolation network
        interpolation_input = torch.cat((previous_frames, next_frames, forward_flow, backward_flow, 
                                       intermediate_backward_flow, intermediate_forward_flow, 
                                       warped_next_frame, warped_previous_frame), dim=1)
        interpolation_output = interpolate(interpolation_input)

        # Refine optical flows and compute visibility maps
        refined_forward_flow = interpolation_output[:, :2, :, :] + intermediate_forward_flow
        refined_backward_flow = interpolation_output[:, 2:4, :, :] + intermediate_backward_flow
        visibility_map_previous = sigmoid(interpolation_output[:, 4:5, :, :])
        visibility_map_next = 1 - visibility_map_previous

        # Final backward warping with refined flows
        final_warped_previous = backward_warping(previous_frames, refined_forward_flow)
        final_warped_next = backward_warping(next_frames, refined_backward_flow)

        # Blend warped frames using visibility maps
        interpolated_frame = ((1 - time_ratio) * visibility_map_previous * final_warped_previous + 
                              time_ratio * visibility_map_next * final_warped_next) / \
                             ((1 - time_ratio) * visibility_map_previous + 
                               time_ratio * visibility_map_next)

        interpolated_frames.append(interpolated_frame)

    return interpolated_frames


def denormalize_frame(tensor_frame, original_width, original_height):
    """Convert tensor frame back to numpy array for video writing."""
    cpu_frame = tensor_frame.cpu()
    pil_frame = tensor_to_image(cpu_frame)
    resized_frame = pil_frame.resize((original_width, original_height), Image.Resampling.BILINEAR)
    rgb_frame = resized_frame.convert('RGB')
    bgr_frame = np.array(rgb_frame)[:, :, ::-1].copy()  # RGB to BGR for OpenCV
    return bgr_frame


def convert_video(input_path, output_path, interpolation_factor, batch_size=10, output_format='mp4v', output_fps=30):
    """Convert video by interpolating frames to increase frame rate."""
    global backward_warping

    video_input = cv2.VideoCapture(input_path)
    total_frame_count = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
    original_width, original_height = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_codec = cv2.VideoWriter_fourcc(*output_format)
    video_output = cv2.VideoWriter(output_path, video_codec, float(output_fps), (original_width, original_height))

    # Ensure dimensions are multiples of 32 for network compatibility
    network_width, network_height = (original_width // 32) * 32, (original_height // 32) * 32
    backward_warping = model.backWarp(network_width, network_height, device).to(device)

    processed_frames = 0
    frame_batch = []
    while True:
        frame_batch = load_batch(video_input, batch_size, frame_batch, network_width, network_height)
        if len(frame_batch) == 1:
            break
        processed_frames += len(frame_batch) - 1

        intermediate_frames = interpolate_batch(frame_batch, interpolation_factor)
        intermediate_frames = list(zip(*intermediate_frames))

        for frame_idx, intermediate_frame_list in enumerate(intermediate_frames):
            video_output.write(denormalize_frame(frame_batch[frame_idx], original_width, original_height))
            for intermediate_frame in intermediate_frame_list:
                video_output.write(denormalize_frame(intermediate_frame, original_width, original_height))

        try:
            yield len(frame_batch), processed_frames, total_frame_count
        except StopIteration:
            break

    video_output.write(denormalize_frame(frame_batch[0], original_width, original_height))

    video_input.release()
    video_output.release()


@click.command('Evaluate Model by converting a low-FPS video to high-fps')
@click.argument('input')
@click.option('--checkpoint', help='Path to model checkpoint')
@click.option('--output', help='Path to output file to save')
@click.option('--batch', default=2, help='Number of frames to process in single forward pass')
@click.option('--scale', default=4, help='Scale Factor of FPS')
@click.option('--fps', default=30, help='FPS of output video')
def main(input, checkpoint, output, batch, scale, fps):    
    load_models(checkpoint)
    start_time = time()
    frame_count = 0
    average_fps = 0
    
    for batch_size, frames_done, total_frames in convert_video(input, output, int(scale), int(batch), output_fps=int(fps)):
        average_fps, frame_count = calculate_average_fps(average_fps, frame_count, batch_size / (time() - start_time))
        progress_percentage = int(100 * frames_done / total_frames)
        estimated_time_remaining = (total_frames - frames_done) / average_fps
        print('\rDone: {:03d}% FPS: {:05.2f} ETA: {:.2f}s'.format(progress_percentage, average_fps, estimated_time_remaining) + ' '*5, end='')
        start_time = time()


if __name__ == '__main__':
    main()

