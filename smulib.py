# Author : Nergo; Contributor : Scott T Keane, Cambridge University

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
import load_intan_rhs_format
from scipy.fft import fftfreq

# Matplotlib backend management

mpl.use("TkAgg")
plt.style.use('dark_background')


# Math functions

def sine(point, offset, amp, points_per_cycle):
    return offset + amp * np.sin(2 * np.pi * (point % points_per_cycle) / points_per_cycle)


# Function for running fits of measured data to get ∆Ig and ∆Id
def sine_fit(t, amp, phi, I0, f_freq):
    return I0 + amp * np.sin(2 * np.pi * f_freq * t + phi)


# CSV handling functions

def _read_csv(name, columns=None, time_array_included=None, skip=0):
    if columns is None:
        columns = ["Volt"]
    if time_array_included is None:
        df = pd.read_csv(name, names=columns, skiprows=skip, low_memory=False)
    else:
        df = pd.read_csv(name, names=columns, index_col=0, skiprows=skip, low_memory=False)
    df.astype('float').dtypes  # Important to be able to apply operations to the dataframe!
    return df


# Uses the filename and columns name as input. Returns a dataframe

# Intan files handling function(s)

def rhs_to_csv(file, file_out):
    data = load_intan_rhs_format.read_data(file)
    val = np.array(data["amplifier_data"])
    np.savetxt(file_out, val[0], delimiter=",")


def input_signal_repositionning(file, file_out_path, list_low_values=[0.001, 0.002],
                                list_step_values=[0.005, 0.004, 0.006, 0.003], columns=None, time_array_included=None):
    start_time = time.time()
    nb_channel = 1
    cu_df = _read_csv(file, columns, time_array_included)
    fig, ax = plt.subplots()
    df_clipped = _clip(cu_df)
    for i in range(len(list_low_values)):
        for j in range(len(list_step_values)):
            df_normalized = _normalize(df_clipped, list_low_values[i], list_step_values[j])
    df_normalized.to_csv(
        file_out_path + '/repositionned_signal_' + str(list_low_values[i]) + str(list_step_values[j]) + '.csv')
    print("Process signal repositionning took\n--- %s seconds ---" % (time.time() - start_time))


# Dataframe operation

def _normalize(df, low, step):
    df_min_max_scaled = df.copy()
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (((df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
                df_min_max_scaled[column].max() - df_min_max_scaled[column].min())) * step) + low
    return df_min_max_scaled


def _clip(df):
    df.clip(lower=-300, upper=300, inplace=True)
    return df


def _plot(df, i, ax1):
    df[str(i)].plot(linestyle='dashed', marker='o', color='y', ax=ax1)
    plt.show()


# List generating functions

def _generate_sweep_from_pd(df):
    sweep = ""
    for i in range(df["Volt"].idxmin(), df["Volt"].idxmax()):
        sweep += "{:.3E}".format(df["Volt"][i], 3) + ","
    sweep += "{:.3E}".format(df["Volt"][df["Volt"].idxmax()], 3)
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
            k = points_per_sweep - j
            sweep += "{:.3E}".format(round(v_start + (k / points_per_sweep) * (v_end - v_start), 5)) + ","
        sweep += "{:.3E}".format(round(v_start, 5))
    return sweep


def _generate_pulse(v_start, v_end, width, time):
    v_list = ""
    points = round(width / time)
    for i in range(0, points):
        v_list += "{:.3E}".format(v_start) + ","
    for j in range(0, points):
        v_list += "{:.3E}".format(v_end) + ","
    for k in range(0, points - 1):
        v_list += "{:.3E}".format(v_start) + ","
    v_list += "{:.3E}".format(v_start)
    return v_list


def _generate_freqs(start_freq, end_freq, points_per_dec):
    decs = np.subtract(np.log10(end_freq), np.log10(start_freq))
    freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), int(decs * points_per_dec))
    meas_time = "{:.1E}".format(1 / (12 * freqs[0]))
    for i in range(1, len(freqs)):
        meas_time += "," + "{:.1E}".format(1 / (12 * freqs[i]))
    meas_time = np.asarray([float(i) for i in meas_time.split(',')])
    freqs = 1 / (12 * meas_time)
    return freqs, meas_time


def _generate_freqs_V2(start_freq, end_freq, points_per_dec, max_meas_time=1e-3, min_meas_time=2e-5):
    decs = np.subtract(np.log10(end_freq), np.log10(start_freq))
    freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), int(decs * points_per_dec))
    meas_time = np.empty((len(freqs),))
    points_per_cycle = np.empty((len(freqs),))
    meas_time[0] = max_meas_time
    for i in range(0, len(freqs)):
        if i > 0:
            meas_time[i] = meas_time[i - 1]
        points_per_cycle[i] = np.round(1 / (freqs[i] * meas_time[i]))
        while points_per_cycle[i] < 100 and meas_time[i] > min_meas_time:
            meas_time[i] = float("{:.1E}".format(meas_time[i] / 2))
            points_per_cycle[i] = np.round(1 / (freqs[i] * meas_time[i]))
        if meas_time[i] < min_meas_time:
            meas_time[i] = min_meas_time
            points_per_cycle[i] = np.round(1 / (freqs[i] * meas_time[i]))
        freqs[i] = 1 / (meas_time[i] * points_per_cycle[i])
    return freqs, points_per_cycle, meas_time


def _generate_sine_sweep(cycles, points_per_cycle, amp, offset):
    sweep = ""
    for i in range(0, int(points_per_cycle * cycles)):
        sweep += "{:.3E}".format(round(sine(i, offset, amp, points_per_cycle), 5)) + ","
    sweep += "{:.3E}".format(round(sine(i + 1, offset, amp, points_per_cycle), 5))
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
            df = _read_csv(file, skip=1)
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
    np.savetxt(path_to_project + "/sweep.csv", transfer_final, delimiter=",")
    print("Process finished")
    print("--- %s seconds ---" % (time.time() - start_time))


def find_Id_range(inst, Vd, Vg, Ig_range):
    inst.write(":sour1:volt:lev:imm " + "{:.3E}".format(Vg))
    inst.write(":sour2:volt:lev:imm " + "{:.3E}".format(Vd))

    inst.write(":outp1 on")
    inst.write(":outp2 on")

    time.sleep(1)

    inst.write(":sens2:curr:aper:auto on")
    inst.write(":sens2:curr:prot 1")
    inst.write(":sens2:func \"curr\"")

    Id_range = 1e-7
    overload = True

    while overload == True:
        inst.write(":sens2:curr:rang " + '{:.0E}'.format(Id_range))
        data_out = float(inst.query(":meas:curr? (@2)"))
        if data_out == np.nan or abs(data_out) > 0.9 * Id_range:
            Id_range = Id_range * 10
        else:
            overload = False

    return Id_range


def _run_list_sweep(inst, parameters, ch1_list, ch2_list, wait=True):
    # parameters format (list of strings, units of seconds, amps, volts):
    # [(0) measurment points,(1) time per point,(2) measurement delay,(3) aquisition time,
    #  (4) gate current range,(5) drain current range,(6) gate points,(7) drain points,
    #  (8) premeasurement voltage hold time]

    inst.write(":sour1:volt:lev:imm " + ch1_list.split(',')[0])
    inst.write(":sour2:volt:lev:imm " + ch2_list.split(',')[0])

    inst.write(":outp1 on")
    inst.write(":outp2 on")

    if wait == True:
        time.sleep(parameters[8])

    # Sets the measurement list of voltages for the gate (channel 1)
    inst.write(":sour1:func:mode volt")
    inst.write(":sour1:volt:mode list")
    inst.write(":sour1:list:volt " + ch1_list)
    # Sets the measurement list of voltages for the drain (channel 2)
    inst.write(":sour2:func:mode volt")
    inst.write(":sour2:volt:mode list")
    inst.write(":sour2:list:volt " + ch2_list)

    # Set range and interval for measurement
    inst.write(":sens1:func \"curr\"")
    inst.write(":sens1:curr:rang:auto off")
    inst.write(":sens1:curr:prot " + parameters[4])
    inst.write(":sens1:curr:rang " + parameters[4])

    inst.write(":sens2:func \"curr\"")
    inst.write(":sens2:curr:rang:auto off")
    inst.write(":sens2:curr:rang " + parameters[5])
    inst.write(":sens2:curr:prot " + parameters[5])

    # Source output ranging set to fixed mode
    inst.write(":sour1:volt:rang:auto off")
    inst.write(":sour1:volt:rang 2")
    inst.write(":sour2:volt:rang:auto off")
    inst.write(":sour2:volt:rang 2")
    # Mesurement wait time set to OFF
    inst.write(":sens1:wait off")
    inst.write(":sour1:wait off")
    inst.write(":sens2:wait off")
    inst.write(":sour2:wait off")

    # Set trigger source to the same mode
    inst.write(":trig1:sour tim")
    inst.write("trig1:tim " + parameters[1])
    inst.write("trig1:acq:coun " + parameters[0])
    inst.write(":trig1:acq:del " + parameters[2])
    inst.write("trig1:tran:coun " + parameters[6])
    inst.write(":trig1:tran:del 0")
    inst.write(":trig2:sour tim")
    inst.write("trig2:tim " + parameters[1])
    inst.write("trig2:acq:coun " + parameters[0])
    inst.write(":trig2:acq:del " + parameters[2])
    inst.write("trig2:tran:coun " + parameters[7])
    inst.write(":trig2:tran:del 0")

    # Measurement interval is set to the same value
    inst.write(":sens1:curr:aper " + parameters[3])
    inst.write(":sens2:curr:aper " + parameters[3])

    inst.write(":form:elem:sens curr,time")

    # Runs the measurement
    inst.write(":init (@1,2)")
    # Fetches the measurement data
    t1 = time.time()
    data_out = inst.query(":fetc:arr? (@1,2)")
    t2 = time.time()
    # print(t1-t2)
    # Convert string of data to numpy array
    data = np.asarray([float(i) for i in data_out.split(',')])
    # Return the measurement results

    # inst.write(":syst:beep 800,0.25")

    data = np.reshape(data, (int(parameters[0]), 4))

    return data


def run_transfer_meas(path, folder, filename, Vg_init, Vg_final, Vd_init, Vd_final, Vd_step, time_per_point,
                      points_per_sweep, reverse_sweep=True, Ig_range=1e-3, Id_range=1e-1, t_hold_before_meas=2,
                      serial='USB0::0x0957::0xCE18::MY51143745::INSTR',
                      instrumentref="Keysight Technologies,B2902A,MY51143745,3.4.2011.5100\n"):
    # Put same value for Vd_init and Vd_final for 1 measurement
    # For Vd_step, don't use 0
    # Ig_range and Id_range  Must be 1e-X format, minimum X = 0 maximum X = 7
    # The folder must be created beforehand
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
    Vd_values = np.empty((round((Vd_final - Vd_init) / Vd_step) + 1,))
    for i in range(0, len(Vd_values)):
        if Vd_init > Vd_final:
            step = -1 * abs(Vd_step)
        else:
            step = abs(Vd_step)
        Vd_values[i] = round(Vd_init + step * i, 4)

    # Generate sweep for gate voltage based on user input
    vg_list = _generate_lin_sweep(points_per_sweep, Vg_init, Vg_final, reverse=reverse_sweep)
    vd_list = "{:.1E}".format(Vd_values[0])

    # parameters format (list of strings, units of seconds, amps, volts):
    # [(0) measurment points,(1) time per point,(2) measurement delay,(3) aquisition time,
    #  (4) gate current range,(5) drain current range,(6) gate points,(7) drain points,
    #  (8) pre-measurement voltage hold time]

    parameters = []
    parameters.append("{:.0f}".format(len(vg_list.split(','))))
    parameters.append("{:.1E}".format(time_per_point))
    parameters.append("{:.1E}".format(time_per_point / 2))
    parameters.append("{:.1E}".format((time_per_point / 2) * 0.8))
    parameters.append("{:.0E}".format(Ig_range))
    parameters.append("{:.0E}".format(Id_range))
    parameters.append("{:.0f}".format(len(vg_list.split(','))))
    parameters.append("{:.0f}".format(len(vd_list.split(','))))
    parameters.append(t_hold_before_meas)

    # Create empty array to store output data
    transfer = np.empty((len(Vd_values), int(parameters[0]), 4))

    # Run transfer curve for each drain voltage and plot results
    # Run output curve for each gate voltage and plot results
    transfer[0] = _run_list_sweep(parameters, vg_list, vd_list)
    Vg = np.asarray([float(i) for i in vg_list.split(',')])
    plt.plot(Vg, transfer[0, :, 2], label="$V_{DS}$ = " + str(Vd_values[0]))

    for i in range(1, len(transfer)):
        vd_list = "{:.1E}".format(Vd_values[i])
        transfer[i] = _run_list_sweep(inst, parameters, vg_list, vd_list)
        plt.plot(Vg, transfer[i, :, 2], label="$V_{DS}$ = " + str(Vd_values[i]))

    plt.gca().invert_yaxis()
    plt.xlabel("$V_{GS}$ (V)")
    plt.ylabel("$I_{DS}$ (A)")
    plt.title("Transfer curves")
    plt.legend()

    # Turn SMU output off after all measurements
    inst.write(":outp1 off")
    inst.write(":outp2 off")

    # Saving data

    trans_data = pd.DataFrame()
    trans_data["Time (s)"] = transfer[0, :, 1]
    trans_data["Vgs (V)"] = Vg
    for i in range(0, len(transfer)):
        trans_data["Ids (A) at Vd = " + str(Vd_values[i]) + " V"] = transfer[i, :, 2]
        trans_data["Igs (A) at Vd = " + str(Vd_values[i]) + " V"] = transfer[i, :, 0]

    trans_data.to_csv(path + folder + filename)


def run_output_meas(path, folder, filename, Vg_init, Vg_final, Vg_step, Vd_init, Vd_final, Ig_range, Id_range, I_init,
                    I_final, I_step, time_per_point, points_per_sweep, reverse_sweep=True, t_hold_before_meas=2,
                    serial='USB0::0x0957::0xCE18::MY51143745::INSTR',
                    instrumentref="Keysight Technologies,B2902A,MY51143745,3.4.2011.5100\n"):
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
    # Generate array of gate voltage values based on user input
    Vg_values = np.empty((round((Vg_final - Vg_init) / Vg_step) + 1,))
    for i in range(0, len(Vg_values)):
        if Vg_init > Vg_final:
            step = -1 * abs(Vg_step)
        else:
            step = abs(Vg_step)
        Vg_values[i] = round(Vg_init + step * i, 4)

    # Generate sweep for drain voltage based on user input
    vd_list = _generate_lin_sweep(points_per_sweep, Vd_init, Vd_final, reverse=reverse_sweep)
    vg_list = "{:.1E}".format(Vg_values[0])

    # Create measurement parameters
    # parameters format (list of strings, units of seconds, amps, volts):
    # [(0) measurment points,(1) time per point,(2) measurement delay,(3) aquisition time,
    #  (4) gate current range,(5) drain current range,(6) gate points,(7) drain points,
    #  (8) pre-measurement voltage hold time]

    parameters = []
    parameters.append("{:.0f}".format(len(vd_list.split(','))))  # 0
    parameters.append("{:.1E}".format(time_per_point))  # 1
    parameters.append("{:.1E}".format(time_per_point / 2))  # 2
    parameters.append("{:.1E}".format((time_per_point / 2) * 0.8))  # 3
    parameters.append("{:.0E}".format(Ig_range))  # 4
    parameters.append("{:.0E}".format(Id_range))  # 5
    parameters.append("1")  # 6
    parameters.append("{:.0f}".format(len(vd_list.split(','))))  # 7
    parameters.append(t_hold_before_meas)  # 8

    # Create empty array to store output data
    output = np.empty((len(Vg_values), int(parameters[0]), 4))

    # Run output curve for each gate voltage and plot results
    output[0] = _run_list_sweep(parameters, vg_list, vd_list)
    Vd = np.asarray([float(i) for i in vd_list.split(',')])
    plt.plot(Vd, output[0, :, 2], label="$V_{GS}$ = " + str(Vg_values[0]))

    for i in range(1, len(output)):
        vg_list = "{:.1E}".format(Vg_values[i])
        output[i] = _run_list_sweep(parameters, vg_list, vd_list)
        plt.plot(Vd, output[i, :, 2], label="$V_{GS}$ = " + str(Vg_values[i]))

    plt.gca().invert_yaxis()
    plt.xlabel("$V_{DS}$ (V)")
    plt.ylabel("$I_{DS}$ (A)")
    plt.title("Output curves")
    plt.legend()

    # Turn instrument output off after measurement
    inst.write(":outp1 off")
    inst.write(":outp2 off")

    I_values = np.empty((round((I_final - I_init) / I_step) + 1,))
    for i in range(0, len(I_values)):
        if I_init > I_final:
            step = -1 * abs(I_step)
        else:
            step = abs(I_step)
        I_values[i] = round(I_init + step * i, 4)

    # Generate sweep for gate voltage based on user input
    I_list = _generate_lin_sweep(points_per_sweep, I_init, I_final, reverse=reverse_sweep)
    # parameters format (list of strings, units of seconds, amps, volts):
    # [(0) measurement points,(1) time per point,(2) measurement delay,(3) acquisition time,
    #  (4) LED points, (5) premeasurement voltage hold time]

    parameters = []
    parameters.append("{:.0f}".format(len(I_list.split(','))))
    parameters.append("{:.1E}".format(time_per_point))
    parameters.append("{:.1E}".format(time_per_point / 2))
    parameters.append("{:.1E}".format((time_per_point / 2)))
    parameters.append("{:.0E}".format(I_list.split(",")))
    parameters.append(t_hold_before_meas)

    # Create empty array to store output data
    transfer = np.empty((len(I_values), int(parameters[0]), 4))

    # Run transfer curve for each drain voltage and plot results
    # Run output curve for each gate voltage and plot results
    transfer[0] = _run_list_sweep(inst, parameters, I_list, False)
    I = np.asarray([float(i) for i in I_list.split(',')])
    plt.plot(I, transfer[0, :, 2], label="$I$ = " + str(I_values[0]))

    for i in range(1, len(transfer)):
        I_list = "{:.1E}".format(I_values[i])
        transfer[i] = _run_list_sweep(parameters, I_list)
        plt.plot(I, transfer[i, :, 2], label="$V$ = " + str(I_values[i]))

    plt.gca().invert_yaxis()
    plt.xlabel("$I$ (V)")
    plt.ylabel("$V$ (A)")
    plt.title("Transfer curves")
    plt.legend()

    # Turn SMU output off after all measurements
    inst.write(":outp1 off")
    inst.write(":outp2 off")

    outp_data = pd.DataFrame()
    outp_data["Time (s)"] = output[0, :, 1]
    outp_data["Vds (V)"] = Vd
    for i in range(0, len(output)):
        outp_data["Ids (A) at Vg = " + str(Vg_values[i]) + " V"] = output[i, :, 2]
        outp_data["Igs (A) at Vg = " + str(Vg_values[i]) + " V"] = output[i, :, 0]

    outp_data.to_csv(path + folder + filename)


# Plotting functions

# Plots transconductance
def plot_transconductance(delta_Id, Vg_amp, freqs):
    gm = np.divide(delta_Id, Vg_amp)
    plt.loglog(freqs, gm)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("gm (S)")
    return gm


# Plots impedance graph

def plot_impedance(Ig_fit, Vg_amp, freqs, points_omitted):
    Z = (np.divide(Vg_amp, Ig_fit[:, 0]))
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (Ohms)', color=color)
    ax1.loglog(freqs, Z, 'x-')
    m, b = np.polyfit(np.log10(freqs[:len(freqs) - points_omitted]), np.log10(Z[:len(Z) - points_omitted]), 1)
    Z_fit = np.power(10, m * np.log10(freqs) + b)
    ax1.loglog(freqs, Z_fit, 'r--')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Phase (deg)', color=color)  # we already handled the x-label with ax1
    phase = Ig_fit[:, 1]
    ax2.semilogx(freqs, phase, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    capacitance = np.divide(np.power(10, -1 * b), 2 * np.pi)
    print("Capacitance = " + str(capacitance) + " F")
    plt.show()
    return capacitance


# Plots curve for mobility fitting
def plot_mobility(Ig_fit, Id_fit, freqs, points_omitted, length, Vd):
    freq_D_Id = np.multiply(Id_fit[:, 0], freqs)[:len(freqs) - points_omitted]
    plt.plot(freq_D_Id, Ig_fit[:len(freqs) - points_omitted, 0])
    m, b = np.polyfit(freq_D_Id, Ig_fit[:len(freqs) - points_omitted, 0], 1)
    plt.plot(freq_D_Id, m * freq_D_Id + b, 'rx--')
    plt.ylabel("∆Ig (A)")
    plt.xlabel("∆Id*freq (A s^-1)")
    tau = np.divide(m, 2 * np.pi)
    L = length * 1e-4
    mu = np.divide(np.square(L), np.multiply(abs(Vd), tau))
    print("Mobility = " + str(mu) + " cm^2 V^-1 s^-1")
    return mu


# Plots pulse transient
def plot_pulse(pulse):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel('I$_{DS}$ (A)')
    ax1.set_xlabel('time (s)', color=color)
    ax1.plot(pulse[:, 1], pulse[:, 2])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('I$_{GS}$ (A)', color=color)  # we already handled the x-label with ax1
    ax2.plot(pulse[:, 1], pulse[:, 0], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
