import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/heath/waypoint_field_nav/install/waypoint_field_nav'
