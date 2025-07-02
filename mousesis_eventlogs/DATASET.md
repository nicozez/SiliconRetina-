# Dataset

The [MouseSIS dataset](https://arxiv.org/pdf/2409.03358) consists of 33 sequences with an approximate duration of 20s each. Each sequences contains pixel-aligned frames and events with accurate time-alignment (hardware-triggered). Additionally, we provide mask-accurate tracking labels for all mice in the sequences (a task we term *spatio-temporal instance segmentation (SIS)*).

The data itself can be found [here](https://drive.google.com/drive/folders/1TQns9-WZw-n26FaUE3gqdAhGgrlRUzCp?usp=drive_link). Each sequence is saved as a .hdf5 file in it's respective split folder. The annotations are provided as .json files in a format similar to the YouTubeVIS format (one per split).

You can comprehend the dataset structure using this visualization script:

```bash
python scripts/visualize_events_frames_and_masks.py --h5_path data/MouseSIS/top/val/seq25.h5 --annotation_path data/MouseSIS/val_annotations.json
```

If you download the whole dataset, the structure of the dataset looks like this.

```txt
data/MouseSIS
├── top/
│   ├── train
│   │   ├── seq_02.hdf5
│   │   ├── seq_05.hdf5
│   │   ├── ...
│   │   └── seq_33.hdf5
|   ├── val
│   │   ├── seq_03.hdf5
│   │   ├── seq_04.hdf5
│   │   ├── ...
│   │   └── seq_25.hdf5
│   └── test
│       ├── seq_01.hdf5
│       ├── seq_07.hdf5
│       ├── ...
│       └── seq_32.hdf5
├── dataset_info.csv
├── val_annotations.json
└── train_annotations.json
```
The .hdf5 files have the following fields ():

```txt
images: (num_images, height, width, 3) uint8
img2event: (num_images,) int64
img_ts: (num_images,) float64  # timestamps in microseconds
p: (num_events,) uint8
t: (num_events,) uint32  # timestamps in microseconds
x: (num_events,) float64
y: (num_events,) float64
```
The field `img2event` is the index of the last event occuring before the start of exposure of an image.

The annotations files have this format:

```json
{
    "info": {
        "description": "string",     // Dataset description
        "version": "string",         // Version identifier
        "date_created": "string"     // Creation timestamp
    },
    "videos": [
        {
            "id": "string",          // Video identifier (range: "01" to "33")
            "width": integer,        // Frame width in pixels (1280)
            "height": integer,       // Frame height in pixels (720)
            "length": integer        // Total number of frames
        }
    ],
    "annotations": [
        {
            "id": integer,           // Unique instance identifier
            "video_id": "string",    // Reference to parent video
            "category_id": integer,  // Object category (1 = mouse)
            "segmentations": [
                {
                    "size": [height: integer, width: integer],  // Mask dimensions
                    "counts": "string"                          // RLE-encoded segmentation mask
                }
            ],
            "areas": [float],        // Object area in pixels
            "bboxes": [              // Bounding box coordinates
                [x_min: float, y_min: float, width: float, height: float]
            ],
            "iscrowd": integer      // Crowd annotation flag (0 or 1)
        }
    ],
    "categories": [
        {
            "id": integer,          // Category identifier
            "name": "string",       // Category name
            "supercategory": "string" // Parent category
        }
    ]
}
```
