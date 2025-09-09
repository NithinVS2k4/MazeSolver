import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import torch
import torch.nn as nn
from MazeEnvironment import MazeEnv


def visualize_cnn(env, cnn, cmap = 'gray'): 
    observations, _ = env.reset()
    img = observations['image']/ 255.0 
    plt.imshow(img)
    plt.show()
    x = torch.FloatTensor(img).unsqueeze(0)
    x = torch.permute(x, (0, 3, 1, 2))
    for layer in cnn:
        with torch.no_grad():
            x = layer(x)
        if isinstance(layer, nn.Conv2d):
            num_channels = x.size()[1]
            fig, axs = plt.subplots(int(round(num_channels**0.5)), int(round(num_channels**0.5)))
            ax = axs.ravel()
            for i in range(num_channels):
                ax[i].set_title(f'Channel {i}')
                ax[i].imshow(x.detach().numpy()[0, i], cmap = cmap)
                ax[i].axis('off')
            plt.show()



def animate_policy(env, model, FPS: int = 12, do_truncate: bool = True, goal_dist=None):
    figure_size = (5, 5)

    s, _ = env.reset(options = {'goal_dist':goal_dist})
    
    env_info = {
        'actions': lambda a: ["↑","→","←","↓"][a[0]],
        'state_interpreter': lambda s: str(s['telemetry']),
    }

    step = 0
    
    while True:
        start_time = time.time()
        
        action = model.predict(s)
        
        step += 1
        
        clear_output(wait=True)
        
        plt.figure(figsize=figure_size)
        plt.imshow(s['image'])
        plt.axis('off')
        
        interp = env_info['state_interpreter'](s)
        action_str = env_info['actions'](action)
        
        # Add information below the image
        plt.text(0.5, -0.15, f"State: {interp}\nAction: {action_str}\nTime Step: {step}", 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='bottom', horizontalalignment='center')
        
        
        plt.show()
        
        s, r, terminated, truncated, _ = env.step(action[0])
        r = float(r)
        
        end_time = time.time()
        if FPS:
            time.sleep(max(0,1 / FPS - (end_time - start_time)))
            
        if terminated or (truncated and do_truncate):
            break
    
    # Show final frame
    clear_output(wait=True)
    img, telemetry = s
    frame = img
    
    plt.figure(figsize=figure_size)
    plt.imshow(s['image'])
    plt.axis('off')
    
    interp = env_info['state_interpreter'](s)
    
    plt.text(0.5, -0.15, f"Final State: {interp}\nAction: {action_str}\nTime Step: {step}", 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='bottom', horizontalalignment='center')
    
    plt.show()




