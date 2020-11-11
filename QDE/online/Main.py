import device_run as device_run
import math
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

model = device_run.initiate()
list_of_gates = [-1097.19870146, -1972.54652661, -1025.76251066,  -964.9129052,
 -1595.64035536, -1037.70755352,  -112.49633387]
# regime 2 [-1237.59158635, -1812.15773984, -1150.96971704, -471.5743834,  -1325.37436217, -1036.47802564,  -635.62973587]
# regime 1 [-1150.91240972, -1947.66738027, -1050, -1172.60790138, -1149.99759364,  -1050, -837.49978806 ]

gates = ["c3", "c4", "c5", "c6", "c7", "c9", "c10"]
gate_numbers = [3, 4, 5, 6, 7, 9, 10]
initial_gate_cu_cv_num_indices = (2, 5)
MaxStep = 300
epsilon = 0.0

pygor_xmlip = 'http://129.67.86.107:8000/RPC2'

starting_positions = [[12, 17],
       [18, 14],
       [ 3,  0],
       [11, 16],
       [17, 20],
       [16, 18],
       [14,  8],
       [ 4, 14],
       [14,  9],
       [ 2,  8]]


time = []
rand_time = []

steps = []
rand_steps = []

for starting_position in starting_positions:
    print('device run, ', starting_position)

    epsilon = 0.0
    savefilename = 'benchmark/regime3_epsilon_'+str(int(epsilon))+'_starting_position_'+str(starting_position[0])+'_'+str(starting_position[1])
    #savefilename = None
    env, run_information = device_run.run(model, epsilon, gates, gate_numbers, list_of_gates, initial_gate_cu_cv_num_indices, pygor_xmlip = pygor_xmlip, show_log = False, MaxStep=MaxStep, starting_position=starting_position,random_pixel = True, savefilename = savefilename)
    time.append(run_information["Total tuning time (seconds)"])
    steps.append(run_information["Number of steps"])

    epsilon = 1.1
    #savefilename = None
    savefilename = 'benchmark/regime3_epsilon_'+str(int(epsilon))+'_starting_position_'+str(starting_position[0])+'_'+str(starting_position[1])
    env, run_information = device_run.run(model, epsilon, gates, gate_numbers, list_of_gates,
                                          initial_gate_cu_cv_num_indices, pygor_xmlip=pygor_xmlip, show_log=False,
                                          MaxStep=MaxStep, starting_position=starting_position, random_pixel=True,
                                          savefilename=savefilename)
    rand_time.append(run_information["Total tuning time (seconds)"])
    rand_steps.append(run_information["Number of steps"])

print('go')
t0 = datetime.now()

centre_voltages = env.block_centre_voltages[math.floor(env.allowed_n_blocks / 2.0)][
    math.floor(env.allowed_n_blocks / 2.0)]
data = env.pygor.do2d(env.control_gates[0], centre_voltages[0] + (env.allowed_n_blocks / 2.0) * env.block_size,
                      centre_voltages[0] - (env.allowed_n_blocks / 2.0) * env.block_size, env.allowed_n_blocks * env.block_size , env.control_gates[1],
                      centre_voltages[1] + (env.allowed_n_blocks / 2.0) * env.block_size,
                      centre_voltages[1] - (env.allowed_n_blocks / 2.0) * env.block_size, env.allowed_n_blocks * env.block_size )

#data = data.data

time_taken = datetime.now() - t0

scan_information = {"Scan data.data": data,
                    "Scan time (s)": time_taken.total_seconds()}

print(scan_information)
pickle_out = open('benchmark/regime_3_full_scan_data_time'+".pickle","wb")
pickle.dump(scan_information, pickle_out)
pickle_out.close()

plt.imshow(data.data[0], extent=[env.block_centre_voltages[0][0][0], env.block_centre_voltages[-1][-1][0],
                         env.block_centre_voltages[-1][-1][1], env.block_centre_voltages[0][0][1]])
plt.colorbar()
plt.title("Full scan")
plt.show()

time_mean = np.mean(time)
rand_time_mean = np.mean(rand_time)

plt.hist([time, rand_time], np.linspace(0, 2500), color=['r', 'b'], label=['Algorithm', 'Random'], alpha=0.6)
plt.title("times ")
plt.xlabel('time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

steps_mean = np.mean(steps)
rand_steps_mean = np.mean(rand_steps)

plt.hist([steps, rand_steps], np.linspace(0, MaxStep), color=['r', 'b'], label=['Algorithm', 'Random'], alpha=0.6)
plt.title("times ")
plt.xlabel('time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

height = [time_taken.total_seconds(), rand_time_mean, time_mean]
error = [[0, rand_time_mean - np.percentile(rand_time,10),time_mean - np.percentile(time,10)],[0, np.percentile(rand_time,90)- rand_time_mean,np.percentile(time,90)- time_mean]]
plt.bar(['Grid Scan', 'Random', 'DRL Algorithm'], height, yerr=error, color=['coral', 'goldenrod', 'lightseagreen'],
        alpha=0.7, ecolor='black', capsize=10)
plt.title('Benchmark time')
plt.ylabel('time (s)')
plt.grid()
plt.savefig('benchmark/basel2_time_performance_hist_region3.png', transparent=True)
plt.show()

height = [env.dim[0]*env.dim[1], rand_steps_mean, steps_mean]
error = [[0, rand_steps_mean - np.percentile(rand_steps,10),steps_mean - np.percentile(steps,10)],[0, np.percentile(rand_steps,90)- rand_steps_mean,np.percentile(steps,90)- steps_mean]]
plt.bar(['Grid Scan', 'Random', 'DRL Algorithm'], height, yerr=error, color=['coral', 'goldenrod', 'lightseagreen'],
        alpha=0.7, ecolor='black', capsize=10)
plt.title('Benchmark steps')
plt.ylabel('Number of steps')
plt.grid()
plt.savefig('benchmark/basel2_steps_performance_hist_region3.png', transparent=True)
plt.show()

device_run.end_session(model)