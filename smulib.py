# Author : Nergo; Contribution : Scott T Keane, Cambridge University

# Import

import glob
import os
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvisa
from scipy import signal
from scipy.fft import fftfreq

# Matplotlib backend management

mpl.use("TkAgg")
plt.style.use('dark_background')


# CSV handling functions

def _read_csv(name, columns=None):
    if columns is None:
        columns = ["Volt"]
    df = pd.read_csv(name, names=columns, index_col=0, skiprows=1, low_memory=False)
    df.astype('float').dtypes  # Important to be able to apply operations to the dataframe!
    return df


# Uses the filename and columns name as input. Returns a dataframe


# List generating functions

def _generate_sweep_from_pd(df):
    sweep = ""
    for i in range(0, len(df["Volt"]) - 1):
        sweep += "{:.3E}".format(df["Volt"][i], 3) + ","
    sweep += "{:.3E}".format(df["Volt"][len(df["Volt"]) - 1], 3)
    print(len(sweep))
    return sweep


# This function takes a panda dataframe as input and use it to generate a list that can be sent to the SMU

def _generate_lin_sweep(points_per_sweep, v_start, v_end, reverse=False):
    sweep = ""
    for k in range(0, points_per_sweep):
        sweep += "{:.3E}".format(round(v_start + (k / points_per_sweep) * (v_end - v_start), 5)) + ","
    sweep += "{:.3E}".format(round(v_start + ((k + 1) / points_per_sweep) * (v_end - v_start), 5))
    if reverse:
        sweep += ","
        for j in range(1, points_per_sweep):
            i = points_per_sweep - j
            sweep += "{:.3E}".format(round(v_start + (k / points_per_sweep) * (v_end - v_start), 5)) + ","
        sweep += "{:.3E}".format(round(v_start, 5))
    return sweep


# This function generate a linear output using a starting value, an ending value and a step value (points per sweep).

# SMU-related functions

def _run_measurement_one_channel(inst, parameters, ch1_list, wait=True, comments=True):
    # parameters format (list of strings, units of seconds, amps, volts):
    # [(0) S1 compliance current,(1) time per point,(2) points per sweep,(3) acquisition time,
    #  (4) premeasurement voltage hold time]
    # Source settings before turn on - the second source is set as shown by carefulness, we wouldn't want to inject a current or voltage in the detector
    inst.write(":sour1:volt:lev:imm " + ch1_list.split(',')[0])
    inst.write(":sens1:volt:prot " + parameters[0])
    # inst.write(":sour2:func:mode curr")
    # inst.write(":sour2:func:lev:imm 0")
    if comments:
        print("SMU settings in progress : 20% \nTurning outputs on ...")
    # Turning outputs on
    inst.write(":outp1 on")
    inst.write(":outp2 on")
    if comments:
        print("Outputs on")

    # If wait is needed (if the SMU was not turned on when you used this function)
    if wait:
        time.sleep(parameters[4])

    if comments:
        print("SMU settings in progress : 30%")
    # Sets the measurement list of voltages (channel 1)
    inst.write(":sour1:func:mode volt")
    inst.write(":sour1:volt:mode list")
    inst.write(":sour1:list:volt " + ch1_list)
    inst.write(":sour2:func:mode amps")
    inst.write(":sour2:curr:lev:imm 0")

    if comments:
        print("SMU settings in progress : 40%")
    # Sense settings
    inst.write(":sens1:func \"curr\"")
    inst.write(":sens2:func \"volt\"")
    inst.write(":sens2:volt:rang:auto on")
    inst.write(":sens1:curr:rang:auto on")
    inst.write(":sens2:curr:prot 2")

    if comments:
        print("SMU settings in progress : 50%")
    # Measurement wait time set to OFF
    inst.write(":sens1:wait off")
    inst.write(":sour1:wait off")
    inst.write(":sens2:wait off")
    # sour2 not touched because we don't really care

    if comments:
        print("SMU settings in progress : 55%")
    # Set trigger source to the same mode
    inst.write(":trig1:sour tim")
    inst.write("trig1:tim " + parameters[1])
    inst.write("trig1:acq:coun " + parameters[2])  # points per sweep
    inst.write(":trig1:acq:del def")
    inst.write("trig1:tran:coun " + parameters[2])
    inst.write(":trig1:tran:del def")
    inst.write(":trig2:sour tim")
    inst.write("trig2:tim " + parameters[1])
    inst.write("trig2:acq:coun " + parameters[2])
    inst.write(":trig2:acq:del def")
    inst.write("trig2:tran:coun " + parameters[2])
    inst.write(":trig2:tran:del def")

    if comments:
        print("SMU settings in progress : 80%")
    # Measurement interval is set to the same value
    inst.write(":sens1:curr:aper:auto on")
    inst.write(":sens2:volt:aper " + parameters[5])

    if comments:
        print("SMU settings in progress : 95%")
    # Output formatting
    inst.write(":form:elem:sens volt,curr,time")

    if comments:
        print("SMU settings in progress : 100% \nRunning measurements ...")
    # Running measurements
    inst.write(":init (@1,2)")

    if comments:
        print("Measurements in progress : Fetching data...")
    # Fetching data - there is a more elegant way to do that using read
    data_raw_input_time = inst.query(":fetc:arr:time? (@1)")
    data_raw_input_volt = inst.query(":fetc:arr:volt? (@1)")
    data_raw_input_curr = inst.query(":fetc:arr:curr? (@1)")
    data_raw_output_time = inst.query(":fetc:arr:time? (@2)")
    data_raw_output_volt = inst.query(":fetc:arr:volt? (@2)")
    data_raw_output_curr = inst.query(":fetc:arr:curr? (@2)")

    if comments:
        print("Data export...")
    # Transforming data from list to array
    data_input_time = np.asarray([float(i) for i in data_raw_input_time.split(',')])
    data_input_volt = np.asarray([float(i) for i in data_raw_input_volt.split(',')])
    data_input_curr = np.asarray([float(i) for i in data_raw_input_curr.split(',')])
    data_output_time = np.asarray([float(i) for i in data_raw_output_time.split(',')])
    data_output_volt = np.asarray([float(i) for i in data_raw_output_volt.split(',')])
    data_output_curr = np.asarray([float(i) for i in data_raw_output_curr.split(',')])
    if comments:
        print("Set of measurement done")
    return data_input_time, data_input_volt, data_input_curr, data_output_time, data_output_volt, data_output_curr


def _data_from_file_measurements(files_list, path_to_project, comp_list, time_before_meas, time_per_point,
                                 instrumentref="Keysight Technologies,B2902A,MY51143745,3.4.2011.5100\n",
                                 time_out=500 * 1e5, serial='USB0::0x0957::0xCE18::MY51143745::INSTR',
                                 file_div=7, wait=True, comments=True):
    for file in files_list:
        folder_results = file[:len(file) - 4]
        path = path_to_project + folder_results
        os.mkdir(path)
        for c in comp_list:
            rm = pyvisa.ResourceManager()
            print(rm.list_resources())
            print("trying" + serial)
            try:
                inst = rm.open_resource(serial)
            except:
                print("Incorrect serial, please modify code")
            assert (inst.query("*IDN?") == instrumentref), \
                print("Houston, we have a problem")
            print("connection successful")
            inst.timeout = time_out  # to change, depends on longest measurement. Not in the readme but pretty obvious
            df = _read_csv(file)
            if df.shape[0] > 2500:  # this is the number of points limit we can send to the SMU
                file_partition = len(df) / file_div
            else:
                file_partition = len(df)
            start_time = time.time()
            print("Beginning", file_partition, "measurements \nfile:" + file)
            folder = "Data_comp_" + str(c) + "\\"
            path2 = os.path.join(path, folder)
            os.mkdir(path2)
            for i in range(2500, int(file_partition), 2500):
                ch1_list = _generate_sweep_from_pd(df[i - 2500:i])

                # parameters format (list of strings, units of seconds, amps, volts):
                # [(0) S1 compliance current,(1) time per point,(2) points per sweep,(3) acquisition time,
                #  (4) premeasurement voltage hold time, (5) measurement delay]
                parameters = []
                parameters.append("{:.0E}".format(c))
                parameters.append("{:.1E}".format((time_per_point)))
                parameters.append("{:.1f}".format(len(ch1_list.split(','))))
                parameters.append("{:.1E}".format(time_per_point / 2))
                parameters.append(time_before_meas)
                parameters.append("{:.1E}".format((time_per_point / 2)))

                # Create empty array to store output data
                transfer = np.empty((len(ch1_list.split(",")), 6), float)
                transfer = _run_measurement_one_channel(inst, parameters, ch1_list, wait, comments)
                transfer[0][:] += (time_per_point * 2500) * (int(i / 2500) - 1)
                transfer_transpose = np.array(transfer)
                transfer_final = np.transpose(np.array(transfer_transpose))

                # Saving CSV
                filename = "SMU_meas_" + str(int(i / 2500)) + ".csv"  # change filename
                np.savetxt(path2 + filename, transfer_final, delimiter=",")
            print("Process finished")
            print("--- %s seconds ---" % (time.time() - start_time))


def _sweep_measurements(path_to_project, v_init, v_final, points_per_sweep, time_before_meas, time_per_point,
                        instrumentref="Keysight Technologies,B2902A,MY51143745,3.4.2011.5100\n",
                        time_out=500 * 1e5, serial='USB0::0x0957::0xCE18::MY51143745::INSTR',
                        wait=True, comments=True, reverse=False):
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    print("trying" + serial)
    try:
        inst = rm.open_resource(serial)
    except:
        print("Incorrect serial, please modify code")
    assert (inst.query("*IDN?") == instrumentref), \
        print("Houston, we have a problem")
    print("connection successful")
    inst.timeout = time_out  # to change, depends on longest measurement. Not in the readme but pretty obvious
    print("Generating Sweep")
    v_list = _generate_lin_sweep(points_per_sweep, v_init, v_final, reverse)
    print("Running measurement set")
    start_time = time.time()
    # parameters format (list of strings, units of seconds, amps, volts):
    # [(0) measurement points,(1) time per point,(2) measurement delay,(3) acquisition time,
    #  (4) LED points, (5) premeasurement voltage hold time]

    parameters = []
    parameters.append("{:.0f}".format(len(v_list.split(','))))
    parameters.append("{:.1E}".format(time_per_point))
    parameters.append("{:.1E}".format(time_per_point / 2))
    parameters.append("{:.1E}".format((time_per_point / 2)))
    parameters.append("{:.0f}".format(len(v_list.split(","))))
    parameters.append(time_before_meas)

    # Create empty array to store output data
    print("Saving CSV file")
    transfer = np.empty((len(v_list.split(",")), 8), float)
    transfer = _run_measurement_one_channel(inst, parameters, v_list, wait, comments)
    transfer_transpose = np.array(transfer)
    transfer_final = np.transpose(transfer_transpose)
    np.savetxt(path_to_project + "/sweep", transfer_final, delimiter=",")
    print("Process finished")
    print("--- %s seconds ---" % (time.time() - start_time))


# Plotting-related functions

def _plotting_csv_files(files_data, name_column_1, name_column_2, label_x1, label_x2, label_y1, label_y2,
                        title, time_per_point=2e-5, notch_bool=False, notch_freq=50,
                        lenfft=100, bandwidth=2, column_1_to_plot='1', column_2_to_plot='3', fft_bool=False):
    files_arr = []
    for dat in files_data:
        folder = dat
        files = os.path.join(folder, "SMU_meas_[0-9].csv")
        files = glob.glob(files)
        files1 = os.path.join(folder, "SMU_meas_[0-9][0-9].csv")
        files1 = glob.glob(files1)
        files = files + files1
        files2 = os.path.join(folder, "SMU_meas_[0-9][0-9][0-9].csv")
        files2 = glob.glob(files2)
        files += files2
        files_arr.append(files)
    df_ex = pd.concat(
        [pd.read_csv(f, names=[str(x) for x in range(5)], index_col=0, skiprows=1, low_memory=False) for f in
         files_arr[0]])
    # #  Timestep
    t = time_per_point
    if notch_bool:
        notch_freq = notch_freq
        quality_factor = notch_freq / bandwidth
        samp_freq = 1 / t
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
        arr = np.abs(
            np.fft.fft(
                signal.filtfilt(b_notch, a_notch, df_ex[column_1_to_plot][1:len(df_ex[column_1_to_plot])].values)))
        data_channel1 = (arr - arr.min()) / (arr.max() - arr.min())
        arr2 = np.abs(np.fft.fft(
            signal.filtfilt(b_notch, a_notch, df_ex[column_2_to_plot][1:len(df_ex[column_2_to_plot])].values)))
        data_channel2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min())
    if fft_bool:
        if not notch_bool:
            data_channel1 = np.fft.fft(df_ex[column_1_to_plot].values)
            data_channel2 = np.fft.fft(df_ex[column_2_to_plot].values)
        fig, axes = plt.subplots(nrows=1, ncols=4)
        # create new x-axis: frequency from signal
        freq_array = fftfreq(len(data_channel1), t)
        axes[0].plot(np.abs(freq_array)[np.abs(freq_array) < lenfft],
                     data_channel1[0:len(np.abs(freq_array)[np.abs(freq_array) < lenfft])] ** 2, label='signal',
                     color='aquamarine')
        axes[1].plot(np.abs(freq_array)[np.abs(freq_array) < lenfft],
                     data_channel2[0:len(np.abs(freq_array)[np.abs(freq_array) < lenfft])] ** 2, label='signal',
                     color='salmon')
        df_ex['1'].plot(color='yellow', label="Detector output", ax=axes[3])
        df_ex['3'].plot(color='limegreen', label="LED output", ax=axes[2])
        axes[0].title.set_text(name_column_1)
        axes[1].title.set_text(name_column_2)
        axes[2].title.set_text("fft" + name_column_1)
        axes[3].title.set_text("fft" + name_column_2)
        axes[0].set_ylabel(label_y1)
        axes[0].set_xlabel(label_x1)
        axes[1].set_ylabel(label_y2)
        axes[1].set_xlabel(label_x2)
        axes[2].set_ylabel("$Power Spectral Density$")
        axes[2].set_xlabel("$Frequency (Hz)$")
        axes[3].set_ylabel("$Power Spectral Density$")
        axes[3].set_xlabel("$Frequency (Hz)$")
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        df_ex[column_1_to_plot].plot(color='yellow', label="Detector output", ax=axes[3])
        df_ex[column_2_to_plot].plot(color='limegreen', label="LED output", ax=axes[2])
        axes[0].title.set_text(name_column_1)
        axes[1].title.set_text(name_column_2)
        axes[0].set_ylabel(label_y1)
        axes[0].set_xlabel(label_x1)
        axes[1].set_ylabel(label_y2)
        axes[1].set_xlabel(label_x2)
    plt.suptitle(title)
    plt.show()
