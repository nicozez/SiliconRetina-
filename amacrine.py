import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
DIRS = [UP, DOWN, LEFT, RIGHT]

#create an LxL image where each pixel is on with a prob. a 
def random_patch(L, a):
    samples = np.random.random(size=(L,L))
    patch = np.zeros((L,L), dtype=int)
    patch[samples < a] = 1
    return patch
#produce a length array of direction codes
def random_moves(T, num_vertices=5):
    vertices = np.sort(np.random.randint(T, size=(num_vertices)))
    vertices = np.insert(vertices, 0, 0)
    moves = list()
    for i in range(T):
        if i in vertices:
            d = np.random.choice(DIRS)
        moves.append(d)
    return np.array(moves)

# compare two successive patches to estimate motion direction 
def predict_dir(patch, patch_prev):
    if np.sum(patch) < 2:
        return -1
    diff_up = np.sum(np.clip(patch[:,1:]-patch_prev[:,:-1],0,None)) / np.sum(patch)
    diff_down = np.sum(np.clip(patch[:,:-1]-patch_prev[:,1:],0,None)) / np.sum(patch)
    diff_left = np.sum(np.clip(patch[:-1,:]-patch_prev[1:,:],0,None)) / np.sum(patch)
    diff_right = np.sum(np.clip(patch[1:,:]-patch_prev[:-1,:],0,None)) / np.sum(patch)
    return np.argmin([diff_up, diff_down, diff_left, diff_right])

#frame is initially 500x500, bg density is 1%, temporal noise is 1/3 of that, 300 frames in the movie , 5 objects
L = 500
a_bg = 0.01
noise_std = a_bg / 3
T = 300

num_shapes = 5
L_shape = 20
a_shape = 0.1

#generate raw video frames with objects 
shape_moves = [random_moves(T) for _ in range(num_shapes)]
shapes = list()
shapes_xy = list()
shapes_xy_max = L-L_shape
for _ in range(num_shapes):
    shapes.append(random_patch(L_shape, a_shape))
    shapes_xy.append((np.random.randint(shapes_xy_max-L_shape*2), np.random.randint(shapes_xy_max-L_shape*2)))

fig, axs = plt.subplots(nrows=1, ncols=num_shapes)
for i in range(num_shapes):
    axs[i].imshow(shapes[i], cmap='gray')
    axs[i].set_title(f'({shapes_xy[i][0], shapes_xy[i][1]})')
plt.show()

#1st animation , raw frames
frames = np.zeros((T,L,L), dtype=int)
bg = np.random.random(size=(L,L))
bg_moves = random_moves(T, num_vertices=3)
for t in range(T):
    bg += np.random.normal(scale=noise_std, size=(L,L))
    if bg_moves[t] == UP:
        bg = np.roll(bg, 1, axis=1)
    elif bg_moves[t] == DOWN:
        bg = np.roll(bg, -1, axis=1)
    elif bg_moves[t] == LEFT:
        bg = np.roll(bg, -1, axis=0)
    elif bg_moves[t] == RIGHT:
        bg = np.roll(bg, 1, axis=0)

    for i in range(num_shapes):
        move = shape_moves[i][t]
        if move == UP:
            shapes_xy[i] = (shapes_xy[i][0], min(shapes_xy[i][1]+1,shapes_xy_max))
        elif move == DOWN:
            shapes_xy[i] = (shapes_xy[i][0], max(shapes_xy[i][1]-1,0))
        elif move == LEFT:
            shapes_xy[i] = (min(shapes_xy[i][0]+1,shapes_xy_max), shapes_xy[i][1])
        elif move == RIGHT:
            shapes_xy[i] = (max(shapes_xy[i][0]-1,0), shapes_xy[i][1])


    frame = np.zeros((L,L), dtype=int)
    frame[bg < a_bg] = 1
    for i in range(num_shapes):
        frame[shapes_xy[i][0]:shapes_xy[i][0]+L_shape,shapes_xy[i][1]:shapes_xy[i][1]+L_shape] = shapes[i]
    frames[t,:,:] = frame


fig, axs = plt.subplots(nrows=1, ncols=1)
axs = [axs]
fig.tight_layout()

def update(i):
    ax = axs[0]
    ax.cla()
    ax.set_aspect('equal')
    ax.imshow(frames[i,L_shape:-L_shape,L_shape:-L_shape], cmap='gray')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

anim = animation.FuncAnimation(fig=fig, func=update, frames=T, interval=1/30*1E3)
plt.show()

#setup amacrine filtering, step through time, compute a predicted motion direction in each AC patch, then apply consensus to decide if patch is 'valid'
L_ac = 20
L_ac_over = 20

cells_xy = np.zeros((int(L/L_ac), int(L/L_ac), 2))
for i in range(cells_xy.shape[0]):
    for j in range(cells_xy.shape[1]):
        cells_xy[i,j,0] = i*L_ac
        cells_xy[i,j,1] = j*L_ac

filtered = np.zeros((T,L,L), dtype=int); filtered[0,:,:] = frames[1,:,:]
states = np.full((T,cells_xy.shape[0],cells_xy.shape[1]), -1)
#compute and consensus-filter motion 
for t in range(1, T):
    for i in range(cells_xy.shape[0]):
        for j in range(cells_xy.shape[1]):
            states[t,i,j] = predict_dir(
                frames[t,i*L_ac:i*L_ac+L_ac_over,j*L_ac:j*L_ac+L_ac_over],
                frames[t-1,i*L_ac:i*L_ac+L_ac_over,j*L_ac:j*L_ac+L_ac_over]
            )
    for i in range(1, cells_xy.shape[0]-1):
        for j in range(1, cells_xy.shape[1]-1):
            consensus = list()
            consensus.extend(states[t,i-1,j-1:j+2])
            consensus.append(states[t,i,j-1])
            consensus.append(states[t,i,j+1])
            consensus.extend(states[t,i+1,j-1:j+2])
            unique, counts = np.unique(consensus, return_counts=True)
            unique = unique[np.argsort(counts)]
            counts = np.sort(counts)
            if unique[-1] == -1:
                filtered[t,i*L_ac:(i+1)*L_ac,j*L_ac:(j+1)*L_ac] = frames[t,i*L_ac:(i+1)*L_ac,j*L_ac:(j+1)*L_ac]
            elif unique[-1] != states[t,i,j] or (len(unique) >= 2 and counts[-2] > 2 and unique[-2] != states[t,i,j]):
                filtered[t,i*L_ac:(i+1)*L_ac,j*L_ac:(j+1)*L_ac] = frames[t,i*L_ac:(i+1)*L_ac,j*L_ac:(j+1)*L_ac]
    print(f'\rFrame {t}/{T}', end='')



fig, axs = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()

def update(i):
    ax = axs[0]
    ax.cla()
    ax.set_aspect('equal')
    ax.imshow(frames[i,L_ac:-L_ac,L_ac:-L_ac], cmap='gray')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = axs[1]
    ax.cla()
    ax.set_aspect('equal')
    ax.imshow(filtered[i,L_ac:-L_ac,L_ac:-L_ac], cmap='gray')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


anim = animation.FuncAnimation(fig=fig, func=update, frames=T, interval=1/30*1E3)
plt.show()
