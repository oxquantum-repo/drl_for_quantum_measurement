import offline_test_run as offline_test_run
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

model = offline_test_run.initiate()

file_name = 'regime_2_full_scan_data_time'

MaxStep = 200
epsilon = 0.0

env, episode_reward, num_steps_in_episode = offline_test_run.run(model,epsilon,file_name, MaxStep=MaxStep, show_log = False, save = False)

print('Number of steps in episode ', num_steps_in_episode)

q = env.where_is_quantum()

offline_test_run.end_session(model)

plt.imshow(env.image)
plt.colorbar()
plt.title(file_name)
plt.show()

plt.imshow(q)
plt.title('q')
plt.show()

plt.imshow(env.pre_classification_prediction)
plt.colorbar()
plt.title("Pre-classifier Prediction")
plt.show()

plt.imshow(env.cnn_prediction)
plt.colorbar()
plt.title("CNN Prediction")
plt.show()

