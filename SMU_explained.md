
# SMU Doc

###### *Special Thanks to Scott T Keene, Cambridge University*
<br/>

# Table of contents
1. [Introduction](#introduction)
2. [Basic syntax](#basicsynt)
3. [Basic commands](#basiccom)
   1. [Query / Statement](#query)
   2. [Connection](#con)
   3. [Source and sense control](#sourcesenc)
   4. [Trigger control](#trig)
   5. [Formatting Output Data](#format)
   6. [Running measurement](#run)
   7. [Fetching the result data](#fetch)
   8. [Run the measurement and fetching data at once](#read)
   9. [Advanced Function](#adv)

### Introduction <a name="introduction"></a>
This project is designed to control a SMU using data points from a csv.

The following lines explains SMU communication syntax (non-exhaustive):

### Basic syntax <a name="basicsynt"></a>

```python
inst.function
```
You are using an *attribute* function of the instrument (inst) you are communicating with.
Here's a short non-exhaustive of the attribute function you can use through the serial:
* write : you are writing to your instrument
* query : you ask something from your instrument and therefore expect an answer

These two attribute functions are all that you need 99% of the time.  
### Basic commands <a name="basiccom"></a>

#### Query / Statement <a name="query"></a>

One may differentiate a query from a statement depending on the presence of a question mark.

Example :
```python
inst.write(":sour1:func:mode curr")
```
and
```python
inst.query(":sour1:func:mode?")
```
The first one is a statement, the second is a query.
#### Connection <a name="con"></a>
```python
serial = 'USB0::0x0957::0xCE18::MY51143745::INSTR'
inst = rm.open_resource(serial)
```
 * defining and connecting to the *inst*rument to use by using a serial. 
 * RM -> Resource Manager, requires pyusb.
<br></br>
#### Source and sense control <a name="sourcesens"></a>
***N.B. These commands can be used for sour1 or sour2 just the same, when one is mentioned***

***N.B. These commands can be used for sens1 or sens2 just the same, when one is mentioned***

```python
inst.write(":sour1:func:mode volt")
```
or
```python
inst.write(":sour1:func:mode curr")
```
* Changes the mode of source output to volt or current.
<br></br>
```python
 inst.write(":sour1:curr:lev:imm 0") 
```
* Changes the output value (here in current mode) to 0. You put the value you are interested in after the imm.
* The mode must have been set ***prior*** to this command. Otherwise, it'll do nothing if the mode of the SMU isn't corresponding to the one mentioned in this command.
<br></br>
```python
inst.write(":outp1 on")
```
or
```python
inst.write(":outp2 on")
```
* This command switches the desired output on (1 or 2).
* If you desire to turn the output off, replace on by off in this command.
* It is advised to switch the level (previous command) before switching the output on. You wouldn't want to fry your device.
<br></br>
*let ch1_list, a sweep list of current value you want your output to give*
```python
inst.write(":sour1:list:curr " + ch1_list)
```
* Sets the list sweep output (current or voltage) data for the specified channel.

:exclamation: The sweep list has a specific format :
* float with same format (Example : format everything to 3e -> scientific notation, 3 digits after the coma )
* Each value separated by coma : yes it's a list

<br></br>

```python
inst.write(":sens1:func:on \"curr\"")
```
or
```python
inst.write(":sens2:func:on \"volt\"")
```
* Enables the specified measurement functions.

:question: To check if it worked, use 
```python
inst.query(":sens1:func?")
```
<br></br>
```python
inst.write(":sens2:curr:rang:auto on")
```
* Enables or disables the automatic ranging function of the specified measurement
channel.

***N.B.*** This is used on both sens and source
<br></br>
```python
inst.write(":sens1:curr:prot 0.4")
```
* Sets the compliance value of the specified channel. The setting value is applied to both positive and negative sides.

***N.B. 1*** This is used on both sens and source

***N.B. 2*** You can specify a different maximum and minimum using :
```python
inst.write(":sens1:curr:prot:neg 0.4")
inst.write(":sens1:curr:prot:pos 0.6")
```
:exclamation: I would advise you to set this after the immediate value of your output and before turning the output on. You can never be too sure.
<br><br/>
```python
inst.write(":sens1:volt:rang 2")
```
* If the automatic ranging function is disabled, this command sets it to 2V : 0 ≤ |V| ≤ 2.12 V.

***N.B.*** This is used on both sens and source
<br><br/>
```python
inst.write(":sens1:wait off")
inst.write(":sour1:wait off")
```
* Measurement wait set to off.

![img.png](img.png)

*Illustration of wait function from Keysight Technologies B2900 Series
Source/Measure Unit command reference*
<br><br/>
```python
inst.write(":sens1:curr:aper " + parameters[3])
```
* Sets the integration time for one point measurement (i.e. It's the acquisition time for one point).

:exclamation: Integration time range : +8E-6 to +2 seconds 
<br><br/>
#### Trigger control <a name="trig"></a>

***N.B. These commands can be used for trig1 or trig2 just the same***

```python
inst.write(":trig1:sour tim")
```
or
```python
inst.write(":trig2:sour tim")
```
* Selects the trigger source (here timer but you can choose another one, please refer to the command reference if need be) for the specified device action (here it's implied it's for all actions on trig1).
* You can assign a specific source for a specific action using a command such as the following for the acquisition:
```python
inst.write(":trig2:acq:sour tim")
```
<br><br/>
```python
inst.write("trig1:tim 1e-1")
```
* Sets the timer at a value being the time per point in seconds.

:exclamation: Trigger TIM minimum value of 2E-5
<br><br/>

```python
inst.write("trig1:acq:coun 100")
```
* Sets the trigger count for the specified device action. For instance here, I will acquire a 100 measurement points.
<br><br/>
```python
inst.write(":trig1:acq:del " + parameters[2])
```
* Sets the trigger delay for the specified device action.

:exclamation: Range delay supported : 0 to 100000 seconds.
<br><br/>

<img src="img_2.png" alt="drawing" width="200"/> <img src="img_1.png" alt="drawing" width="206"/>

<img src="img_3.png" alt="drawing" width="410"/>

*Illustrations of trigger function from Keysight Technologies B2900 Series
Source/Measure Unit command reference*

:bulb: If there is a specific starting condition to start the trigger function, you have to use ARM commands (allows one to use LXI Trigger Events). If I am not mistaken, the ARM condition roughly tells the trigger timer when to start. It is required for more high-level control and specific use. LXI triggering uses UDP Port/TCP Socket Listener
and the LAN Event Sender.

### Formatting Output Data <a name="format"></a>

```python
inst.write(":form:elem:sens curr,time")
```
* Specifies the elements included in the sense or measurement result data.
* Make a list of what you want to include, the exemple here provides current and time. One may add : volt, res[istance], stat[us], sour[ce].

:exclamation: Order of returned data (depending on which you included): voltage, current,
resistance, time, status, source.
<br><br/>
```python
inst.write(":form:elem:calc calc,time")
```
* Specifies the elements included in the calculation result data.

:exclamation: Order of returned data: calc, time, status.

***N.B.*** I didn't mention calculation before, it is basically a way to manipulate the output data using math function. Please refer to the command reference or feel free to complete this document if need be.

### Running the measurement <a name="run"></a>

```python
inst.write(":init (@1,2)")
```
* Initiates the specified device action for the specified channel. Trigger status is changed from idle to initiated.
* Here the command does not specify an action, therefore it initates all and runs the measurement.

:exclamation: This command must be placed after every command mentioned above. It is obvious that you need to set measurement parameters before launching the said measurement.

### Fetching the result data <a name="fetch"></a>

After running your measurement, you have to fetch the resulting data in order to store them in a variable.

```python
data_out = inst.query(":fetc:arr? (@1,2)")
```
* Here you expect to fetch an array of data. It returns the array data which contains all of the voltage measurement data, current measurement data, resistance measurement data, time data, status data, or source output setting data specified by the :FORM:ELEM:SENS command.
* In (@1,2) you specify the channel list where to fetch the data (here channel 1 AND 2). You may use (@1) if you wish to fetch data from channel 1 only, of (@2) channel 2 only.

<br><br/>
If you wish to fetch only a specific measurement, use the following command:
```python
data_current = inst.query(":fetc:arr:curr? (@1,2)")
```
* This exemple is for current, you may replace curr by volt, res... depending on what you need.

<br><br/>
You may also want to fetch a scalar depending on your use.
```python
data_out = inst.query(":fetc:scal? (@1,2)")
```
* Here you expect to fetch a scalar data. It returns the latest voltage measurement data, current measurement data, resistance measurement data, time data, status data, or source output setting data specified by the :FORM:ELEM:SENS command.
* In (@1,2) you specify the channel list where to fetch the data (here channel 1 AND 2). You may use (@1) if you wish to fetch data from channel 1 only, of (@2) channel 2 only.
<br><br/>

Similarly to array commands, you can fetch a specific scalar:
```python
data_out = inst.query(":fetc:scal:curr? (@1,2)")
```
* This exemple is for current, you may replace curr by volt, res... depending on what you need.

:exclamation: If nothing is specified after ":fetc", it will fetch a scalar by default i.e.
```python
data_out = inst.query(":fetc:scal? (@1,2)")
```
is equivalent to
```python
data_out = inst.query(":fetc? (@1,2)")
```


### Run the measurement and fetching data at once <a name="read"></a>

To run the measurement and fetch data with one command, you may use the read function :
```python
inst.query(":read:arr? (@1,2)")
```
or
```python
inst.query(":read:arr:curr? (@1,2)")
```
or
```python
inst.query(":read:scal? (@1,2)")
```
or
```python
inst.query(":read:scal:curr? (@1,2)")
```
By now, you must have understood how it worked.

:exclamation: Similarly to previous mention, if ":scal" or ":arr" is not mentioned, it fetches a scalar by default.

### Advanced function <a name="adv"></a>

You may use advanced function to control the instrument. For instance, you may generate a beep sound of 800Hz frequency during 1.5 seconds (```inst.write(":sys:beep 800, 1.5")```).

You may check the command reference book for further information as it won't be developed here.
