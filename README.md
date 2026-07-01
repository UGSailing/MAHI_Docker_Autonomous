# Start a autonomous race on the physical boat with MAHI
This code builds on the documentation on getting a connection with the MAHI sense at github.com/UGSailing/documentation/edit/main/autonomous/mahi-sense/README_customer_software.md
This also correlates to the ECU statesmachine in the firmware repo: https://github.com/UGSailing/firmware-Gen1/blob/Autonomous-changes/ecu/README.md
The MAHI Sense reads the states of the ECU and updates his as AM is selected and as a race is selected.
The goal is the following:

State:0
Check ECU message (1 hz) to check if AM is on

State:1
Check if camera, GNSS-coordinates and docker shell work properly

State:2
Check the state of the ECU to select a specified race,

State3:
Execute the specific race

State4:
Finished normally, return to state 0

State5:
Finished with errors, return to states 0



### Notes:

Each state is initiated by the succes of the predecessor. (Except for state 5)
Everything is in LSB, so -1=0b00000001 for 1 Byte and, by convention, the last Byte is used first (the previous Bytes aren't used in this version).
The ECU sends messages at 0x211 with a state and selected mission. The selected mission is -1 is there is no race selected.

The (MAHI Sense) message_id is 99 and 19 for error code. The node_id of the MAHI Sense is 3, so every message send, has to fulfill the equation message_id%16=3. Lower IDs have higer priorities hence the lower id for error messages.
The error state is denoted as -1 (instead of 5) and the error code has this syntax:
bit 1: GNSS
bit 2: camera
bit 3: other-software-issue
For example: ... 00000101 means the camera failed

