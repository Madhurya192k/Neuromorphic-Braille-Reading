import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as color
import matplotlib.cm as cmx
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import time, os, json

def main():
  # Set flags for figures
  disp_heatmap = True
  disp_spikes = True
  disp_spikes3d = True
  
  # Font sizes
  TITLE_SIZE = 20
  AXES_SIZE = 16
  TICKS_SIZE = 12

  # Experiment settings
  sensor_name = 'DVXplorer'
  experiment = 'horizontal_shear'
  events_on = True
  events_off = True

  # Directories
  data_dir_name = experiment + '_08061614'
  home_dir = os.path.join(os.environ['DATAPATH'], 'NeuroTac_' + sensor_name, experiment)
  data_dir = os.path.join(home_dir, data_dir_name,'events')
  fig_save_dir = os.path.join(data_dir, 'figures')
  if not os.path.exists(fig_save_dir):
      os.makedirs(fig_save_dir)

  # Set sensor size
  sensor_x = 0
  if sensor_name == 'DAVIS240':
      sensor_x = 240
      sensor_y = 180
  elif sensor_name == 'DVXplorer':
      sensor_x = 640
      sensor_y = 480
  elif sensor_name == 'eDVS':
      sensor_x = 128
      sensor_y = 128
  else:
    print('Error : sensor name not recognised')
    
  # Load metadata
  with open(data_dir + "/meta.json", "r") as read_file:
      meta = json.load(read_file)
  n_poses = len(meta['obj_poses'])

  # # Load timestamps
  # filename = os.path.join(data_dir,'starting_timestamps.pickle')
  # infile = open(filename,'rb')
  # timestamps = pickle.load(infile)

  # Run display loop 
  for pose_idx in range(n_poses):
      for trial_idx in range(meta['n_runs']):

# Extract slide parameters without brackets and quotes
          slide_depth = meta['slide_depths'][0]
          slide_speed = meta['slide_speeds'][0]
          slide_direction = meta['slide_directions'][0]

            # Load data (events on)
          filename = os.path.join(data_dir, f'letter_{pose_idx}_d{slide_depth}_s{slide_speed}_dir_{slide_direction}_r{trial_idx}_events_on')
          with open(filename, 'rb') as infile_on:
              data_on = pickle.load(infile_on)

            # Load data (events off)
          filename = os.path.join(data_dir, f'letter_{pose_idx}_d{slide_depth}_s{slide_speed}_dir_{slide_direction}_r{trial_idx}_events_off')
          with open(filename, 'rb') as infile_off:
              data_off = pickle.load(infile_off)

  #################################################################### HEATMAP #########################################################################################
          if events_on:
            if disp_heatmap:
              heatmap_data = []
              for x in data_on: 
                  for y in x:
                      heatmap_data.append(len(y))
              heatmap_data = np.reshape(heatmap_data,(sensor_x,-1))
              plt.imshow(heatmap_data.T)
              cbar = plt.colorbar()
              cbar.set_label('number of spikes')
              plt.xlabel('x')
              plt.ylabel('y')
              # plt.show()
              plt.title("Heatmap (ON events) - Pose " + str(pose_idx) + " Trial " + str(trial_idx))
              plt.savefig(fig_save_dir + '/heatmap_pose_' + str(pose_idx) + '_trial_' + str(trial_idx) + '_events_on.png')
              plt.clf()

          if events_off:
            if disp_heatmap:
              heatmap_data = []
              for x in data_off: 
                  for y in x:
                      heatmap_data.append(len(y))
              heatmap_data = np.reshape(heatmap_data,(sensor_x,-1))
              plt.imshow(heatmap_data.T)
              cbar = plt.colorbar()
              cbar.set_label('number of spikes')
              plt.xlabel('x')
              plt.ylabel('y')
              # plt.show()
              plt.title("Heatmap (OFF events) - Pose " + str(pose_idx) + " Trial " + str(trial_idx))
              plt.savefig(fig_save_dir + '/heatmap_pose_' + str(pose_idx) + '_trial_' + str(trial_idx) + '_events_off.png')
              plt.clf()

##################################################################### SPIKES #########################################################################################
          if events_on:
            if disp_spikes:
              spikes_data = data_on.flatten()
              neuron_idx = 0

              for spike_train in spikes_data:
                if spike_train != []:
                  # spike_train = [x-timestamps[trial_idx] for x in spike_train]
                  y = np.ones_like(spike_train) * neuron_idx
                  plt.plot(spike_train, y, 'k|', markersize=0.7)
                neuron_idx +=1
              plt.ylim = (0, len(spikes_data))
              plt.ylabel("neuron")
              plt.xlabel("time (ms)")
              plt.title("Spike data (ON events) - Pose " + str(pose_idx) + " Trial " + str(trial_idx))
              plt.setp(plt.gca().get_xticklabels(), visible=True)

              # plt.show()
              plt.savefig(fig_save_dir + '/spikes_pose_' + str(pose_idx) + '_trial_' + str(trial_idx) + '_events_on.png')
              plt.clf()

          if events_off:
            if disp_spikes:
              spikes_data = data_off.flatten()
              neuron_idx = 0

              for spike_train in spikes_data:
                if spike_train != []:
                  y = np.ones_like(spike_train) * neuron_idx
                  plt.plot(spike_train, y, 'k|', markersize=0.7)
                neuron_idx +=1
              plt.ylim = (0, len(spikes_data))
              plt.ylabel("neuron")
              plt.xlabel("time (ms)")
              plt.title("Spike data (OFF events) - Pose " + str(pose_idx) + " Trial " + str(trial_idx))
              plt.setp(plt.gca().get_xticklabels(), visible=True)

              # plt.show()
              plt.savefig(fig_save_dir + '/spikes_pose_' + str(pose_idx) + '_trial_' + str(trial_idx) + '_events_off.png')
              plt.clf()

##################################################################### 3D SPIKES #########################################################################################
          if disp_spikes3d:
            fig = plt.figure(figsize=[10,5])
            ax = fig.add_subplot(projection='3d')
            
            x = []
            y = []
            spike_train = []
            for x_idx in range(len(data_on)): 
              for y_idx in range(len(data_on[x_idx])): 
                if data_on[x_idx][y_idx] != []:
                  spike_train.extend(data_on[x_idx][y_idx])
                  y.extend(np.ones_like(data_on[x_idx][y_idx]) * y_idx)
                  x.extend(np.ones_like(data_on[x_idx][y_idx]) * x_idx)
            
            ax.scatter(x,spike_train,y, marker = '.',s=0.6,c=spike_train, cmap='gnuplot')
            ax.set_xlabel('X')
            ax.set_ylabel('Time (ms)')
            ax.set_zlabel('Y')
            plt.title("3D Spike data (ON events) - Pose " + str(pose_idx) + " Trial " + str(trial_idx))

            # plt.show(fig)
            ax.view_init(elev=5*pose_idx, azim=-90+pose_idx*10)
            plt.savefig(fig_save_dir + '/spikes3d_pose_' + str(pose_idx) + '_trial_' + str(trial_idx) + '_events_on.png')
            plt.close(fig)
  
if __name__ == '__main__':
    main()




