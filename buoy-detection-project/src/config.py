TEST_PADAANPASSING_ZONDER_BOEIEN = True # dit zal gwn het pad eens aanpassen wnr hij in de buurt van eerste boei komt, om te testen hoe boot reageert zonder boeien nodig te hebben


MARGE = 4
STATE_TRANS_DIST = MARGE * 2
BUOY_MATCH_DISTANCE = 7 #MARGE + 1.5

APRIORI_BUOYLIST = [
    [(51.14339735730541, 2.7468965058450774)],   # buoy 0 — replace with real a-priori GPS coords
    [(51.14344451713093, 2.7475114395335662)],   # buoy 1 — replace with real a-priori GPS coords
]

TILT = 0.5



# padplanning interpolatie:

INTERPOLATE_USING_DISTANCE = True  # bool — use WAYPOINT_DISTANCE instead of N_ARC_PTS/N_SLALOM_PTS
WAYPOINT_DISTANCE        = 2.0   # m — spacing between waypoints - - wordt niet gebruikt als INTERPOLATE_USING_DISTANCE false is
N_ARC_PTS = 12           # waypoints voor de 180° bocht om B2 - worden niet gebruikt als INTERPOLATE_USING_DISTANCE true is
N_SLALOM_PTS = 20        # waypoints per slalom-been (S-curve) - worden niet gebruikt als INTERPOLATE_USING_DISTANCE true is



# look ahead (best afstellen in meters)

METER_LOOK_AHEAD  = 10.0                                    # m
INDEX_LOOK_AHEAD  = round(METER_LOOK_AHEAD / WAYPOINT_DISTANCE)



# speed & ramp:

FAST_SPEED = 4
SLOW_SPEED = 2
N_RAMP_METERS = 5.
N_RAMP_WAYPOINTS  = round(N_RAMP_METERS / WAYPOINT_DISTANCE)     # intermediate speed steps at slow↔fast transitions

