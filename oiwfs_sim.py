#!/usr/bin/env python
#
# Simple simulation of IRIS OIWFS probe arm motion using 2d line segments and
# circles for the heads. It is also possible to specify a position and
# rotation of IRIS, star positions in celestial coordinates, and determine
# the projected outline of the probes on the sky.
#
# Classes:
#   Probe - Current orientation of a probe, velocity, and target star
#   State - Manage a collection of probes, including collision avoidance

import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import sys
import time
from astropy import wcs
import random

# Constants
platescale    = 2.182   # platescale in OIWFS plane (mm/arcsec)
r_patrol      = 60*platescale # radius of the patrol area in (mm)
r_max         = 300         # maximum extension of probes (mm)
r_overshoot   = 20          # distance by which probes overshoot centre (mm)
r_head        = 10          # circumscribe 14x14 square (mm)
r_min         = r_patrol    # minimum extension of probes (mm)
r_star        = 11.5*platescale # minimum allowable separation between stars (mm)
r0            = r_max-r_patrol # initial/parked probe extension (mm)
maxstars      = 3     # maximum number of stars
r_ifu         = 20    # radius of region to avoid for IFU (mm)

vmax = 13.2
vr_max        = vmax #vmax/np.sqrt(2)
vt_max        = vmax/r_min#vr_max/r_max
#vt_max_deg    = 1     # maximum rotator actuator velocity (deg/s)
#vr_max        = 2.5   # maximum extender actuator velocity (mm/s)
at_deg        = 3.2   # rotator actuator acceleration (deg/s^2)
ar            = 8.0   # extender actuator acceleration (m/s^2)
tol_col       = 1     # tolerance for probe collisions (mm)

#tol_avoid     = 100    # repulsive potential range (approaches 0 beyond)
#tol_attract   = 5      # attractive potential range (quadratic->conic)
#alpha         = 10.    # strength of collision avoidance potential
#beta          = 0.001  # strength of target attraction potential

tol_avoid     = r_star #100    # repulsive potential range (approaches 0 beyond)
tol_attract   = 5      # attractive potential range (quadratic->conic)
alpha         = 10.    # strength of collision avoidance potential
beta          = 0.001  # strength of target attraction potential

col_min       = 0.1    # closer to collision than this and we clip col potential
vt_tol_deg    = 0.5    # what rotator velocity is stopped? (deg/s)
vr_tol        = 1.25   # what extender velocity can be considered stopped? (mm/s)
dt            = 0.05   # simulation time step size (s)

# derived values
r_origin      = r_max-r_overshoot  # distance of probe origin from centre
#vt_max        = np.radians(vt_max_deg)  # max rotator velocity (rad/s)
vt_tol        = np.radians(vt_tol_deg)  # stop tolerance rotation (rad/s)
at            = np.radians(at_deg)      # angular acceleration (rad/s^2)
# tolerance for arriving at destination (mm) - related to step sizes/vel
tol_stuck     = dt*np.sqrt(vr_max**2 + (vt_max*r_max)**2)
#tol           = 3.*dt*np.sqrt(vr_max**2 + (vt_max*r_max)**2)
tol           = 1
tol_sq        = tol**2
tol_comp      = tol/2 # tangential/radial component of tolerance
tol_col_sq    = tol_col**2
tol_avoid_sq  = tol_avoid**2
theta_max = np.arcsin(r_patrol/(r_max-r_overshoot)) # max rotator offset
theta_max_deg = np.degrees(theta_max)
grad_tol      = 0.1*0.5*beta*vmax*dt # target potential gradient 1/2 step away
                                     # from the minimum
                                 
d_clear       = 2.*r_origin*np.cos(np.radians(30)) # total probe lengths clear

d_limit = 10*r_patrol    # distance for merit calc if probe is in limit for configuration
d_collided = 20*r_patrol # distance for merit calc if probe collides in configuration

#print "Arrival tolerance (mm):",tol

# max value of collision potential
u_col_max     = 0.5*alpha*(1/col_min - 1/tol_avoid)**alpha

print "Max speed linear stage:",vr_max,"mm/s"
print "Max speed rotary stage:",vt_max,"rad/s"

#sys.exit(1)

# sequence of predefined asterisms for testing
aster_easy = [
    [  (-113,  19), ( -22,  44), (22,22) ],
    [  ( -22,  44), (-113,  19), (22,22) ],
    [  ( 127,  25), (  39, -80), (0,0) ],
    [  ( -40,   0), (   0,   0), (-40,-116) ],
    [  ( -67,   6), ( -86,  35), (  0, 0) ],
    [  (-113,  19), ( -85, -77), (  0, 0) ],
    [  (0,0), (-50, 55), (22,22) ],
    [ (15, 5),  (10, -40), (-10, -5) ],
    [ (50,0), (-29,53), (-29,-53) ],
    [ (-50,0), (29,-53), (29,53) ],
    [ (50,0), (-29,53), (-29,-53) ],
    [ (-50,0), (29,-53), (29,53) ],
    [ (10, 5),(10, -65), (-10,-25) ],
    [ (10, 5),(10, -65), (-20,-95) ],
    [ (10, 5),(10, -65), (-10,-25) ],
    [ (10, 5),(10, -65), (-20,-95) ],
    [ (10, 5),(-20,-75), (10, -65) ]
]


aster_seq = [
    #[  (-113,  19), ( -22,  44), (22,22) ],
    #[  ( -22,  44), (-113,  19), (22,22) ],
    [ (15, 5),  (10, -40), (-10, -5) ],
    [  ( 127,  25), (  39, -80), (0,0) ],
    [  ( -40,   0), (   0,   0), (-40,-116) ],
    #[  ( -67,   6), ( -86,  35), (  0, 0) ],
    #[  (-113,  19), ( -85, -77), (  0, 0) ],
    #[  (0,0), (-50, 55), (22,22) ],
    #[ (15, 5),  (10, -40), (-10, -5) ],
    [ (50,0), (-29,53), (-29,-53) ],
    [ (-50,0), (29,-53), (29,53) ],
    [ (50,0), (-29,53), (-29,-53) ],
    [ (-50,0), (29,-53), (29,53) ],
    [ (10, 5),(10, -65), (-10,-25) ],
    [ (10, 5),(10, -65), (-20,-95) ],
    [ (10, 5),(10, -65), (-10,-25) ],
    [ (10, 5),(10, -65), (-20,-95) ],
    [ (10, 5),(-20,-75), (10, -65) ]
]

aster_hard = [
#    [  (0,0), (-50, 55), (22,22) ],
#    [ (10, 5),  (10, -30), (-10, -5) ],
    [ (10, 5),(10, -65), (-20,-35) ],
    [ (10, 5),(10, -65), (-20,-95) ],
    [ (10, 5),(10, -65), (-20,-35) ],
    [ (10, 5),(10, -65), (-20,-95) ],
    [ (10, 5),(-20,-75), (10, -65) ]
]

aster_test = [
    [ (-60.4716091079,40.6646902121), (-78.5700361149,-85.1306245579), (14.189597544,-9.54389074502) ],
    [ (125.421330453,21.0028940272), (-43.2227332253,59.4616554138), (96.6688013784,-5.84546652608) ]
]

aster_move = [
    [ ( -20,  -5), (   0,-125), ( -20, -100) ],
    [ (  25,  -5), (   0,-125), ( -30, -125) ],
]

#aster = aster_easy
#aster = aster_hard
#aster = aster_test



#    [(32.441075832954759, 18.70708893883236), (-32.768463744706828, -63.266244134945651), (11.308533443719961, -62.810064169565329)],
#    [(-48.991516907841159, 52.321497479289668), (-50.693027314231173, 20.300356482138916), (51.444005809984198, 29.210834003396563)],
#    [(-48.256777829714217, 42.373262727054211), (-67.962669005166447, -4.9640355878341964), (26.190550838561283, 21.220271209887514)],
#    [(5.7629247383486737, 2.037914895370581), (17.107650039880113, -69.834774799832928), (70.908132277294683, -52.643458365061718)],
#    [(11.730895644703848, 50.67128018560495), (-8.5549019694122617, -59.167273554749329), (54.223170263864773, 23.069962822580301)],
#    [(35.85823001832911, 24.128818198711414), (-77.341426873757356, 4.3115710911864831), (-9.2219294870486515, -21.020783354016711)],


# --- Helper functions/classes -------------------------------------------------

# Check for counter-clockwise. This function, and line segment intersection
# algorithm that calls it, are from here:
#  http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# A 2d point
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Calculate the nearest point (and distance) on a line segment AB to a point C.
# http://paulbourke.net/geometry/pointlineplane/
def nearest_linesegment_point(A,B,C):
    if (A.x == B.x) and (A.y == B.y):
        N = Point(A.x,A.y)
        d_sq = (C.x-A.x)**2 + (C.y-A.y)**2

    else :
        u = ((C.x-A.x)*(B.x-A.x) + (C.y-A.y)*(B.y-A.y)) / \
            ((B.x-A.x)**2 + (B.y-A.y)**2)

        if (u < 0) or (u > 1):
            # Closest point not in segment, so choose distance to nearest
            # endpoint
            all_d_sq = [(C.x-A.x)**2 + (C.y-A.y)**2,
                        (C.x-B.x)**2 + (C.y-B.y)**2]
            d_sq = min(all_d_sq)

            if d_sq == all_d_sq[0]:
                N = A
            else:
                N = B
        else:
            # coordinates of the intersection with line segment, the
            # nearest point
            N = Point(A.x + u*(B.x-A.x), A.y + u*(B.y-A.y))

            # squared distance between points
            d_sq=(C.x-N.x)**2 + (C.y-N.y)**2

    return N, d_sq

# Calculate the distance between a line (y = mx + b) and a point P
# Modified from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
def dist_line_point(m,b,P):
    d = np.abs(-m*P.x + P.y - b)/np.sqrt(m**2 + 1)
    return d

# a star
class Star(Point):
    def __init__(self, *args, **kwargs):
        super(Star, self).__init__(*args, **kwargs)

        # display element for star that will be animated when updated
        self.symbol = None

        # catalog entry index if relevant
        self.catindex = None

    def update_symbol(self):
        """ Update the star symbol """
        self.symbol.set_data([self.x], [self.y])

# Exception class for probe actuators at limits
class ProbeLimits(Exception):
    pass

# Exception class for collision
class ProbeCollision(Exception):
    pass

# Exception class for probe vignetting IFU
class ProbeVignetteIFU(Exception):
    pass

# Calculate acceleration along axis to reach target
def calc_accel( x, v, v_target, delta_x, amax, info=False ):
    """ Figure out acceleration

    x -- current coordinate
    v -- current velocity
    v_target -- target velocity
    delta_x -- distance to target
    amax -- magnitude of maximum acceleration
    """

    delta_v = v_target - v
    vsign = np.sign(delta_v)
    dsign = np.sign(delta_x)
    if delta_x != 0:
        a_stop = v**2/(2*abs(delta_x))
    else:
        a_stop = 0

    if a_stop >= 0.75*amax:
        # Start deceleration if we're getting close.
        # Allow a_stop values that vary from amax to
        # allow for roundoff corrections by the acceleration
        if a_stop > 1.5*amax:
            a = -1.5*amax*dsign
            foo='A'
        elif a_stop < 0.5*amax:
            a = -0.5*amax*dsign
            foo='B'
        else:
            a = -a_stop*dsign
            foo='C'

        if info:
            print "calc_accel: stopping",a,amax,dsign,a_stop,foo

    else:
        # Otherwise accelerate towards target velocity
        a = delta_v/dt

        if abs(a) > amax:
            a = amax*vsign

        if info:
            print "calc_accel: acceleration",a,amax,a_stop,v,v_target,vsign


    return a

# Scale a directional polar coordinate velocity vector to maximum speed.
def max_vel(dir):
    dir_abs = np.fabs(dir)
    if dir_abs[0] < 1e-6:
        speed = vt_max/dir_abs[1]
    elif dir_abs[1] < 1e-6:
        speed = vr_max/dir_abs[0]
    else:
        speed = min([vr_max/dir_abs[0],vt_max/dir_abs[1]])
        #norm = np.linalg.norm(dir * [1)
        #speed = vmax/norm

    #print "Vel cal:",dir,speed

    return dir*speed

# --- class Probe --------------------------------------------------------------

class Probe(object):
    def __init__(self, x0, y0, r, theta_home, r_head=r_head,):
        """ Initialize a probe arm

        x0 -- x-coordinate of arm origin
        y0 -- y-coordinate of arm origin
        r -- radius of arm extension actuator
        theta_home -- rotation arm actuator angle home (radians)
        r_head -- radius of head. So we can set to 0 for special tests.
        """

        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.theta_home = theta_home
        self.theta = self.theta_home
        self.r_head = r_head
        self.moving = None  # None, 'moving'

        # Keep track of the last step to see if we're stuck
        self.last_delta = [None,None]

        # calculate initial x, y for probe tip
        self.x, self.y = self.pol2cart(self.r, self.theta)

        # Velocities and accelerations for each axis
        self.vr = 0
        self.vt = 0
        self.ar = 0
        self.at = 0

        # Reference to selected star, and the stars polar coordinates
        self.star = None
        #self.star_r = None
        #self.star_theta = None

        # display elements for probe that will be animated
        self.line = None
        self.trail = None
        self.trail_x = []
        self.trail_y = []
        self.head = None

        # Calculate the home/parked position
        self.x_home,self.y_home = self.pol2cart(r0,self.theta_home)

        # Set if probe is parked/parking
        self.park = False

    def set_cart(self, x, y):
        """ Set probe tips to new x, y location """
        self.x = x
        self.y = y
        self.r, self.theta = self.cart2pol(self.x, self.y)
        self.check_ranges()

    def set_pol(self, r, theta):
        """ Set probe tips to new r, theta location """
        self.r = r
        self.theta = theta
        self.x, self.y = self.pol2cart(self.r, self.theta)
        self.check_ranges()

    def dist(self, probe2, info=False):
        """ Find dist_sq, endpoints of shortest line between two probes.
            If dist_sq == 0 they have collided

        return:

        d_sq -- squared distance to probe2
        A    -- nearest point on this probe
        B    -- nearest point on probe2
        """

        # The origins of the probes will never be the nearest points, so
        # just check the probe tips with the other line segment

        origin = Point(self.x0, self.y0)
        tip = Point(self.x, self.y)
        rh = self.r_head
        origin2 = Point(probe2.x0, probe2.y0)
        tip2 = Point(probe2.x, probe2.y)
        rh2 = probe2.r_head

        N1, d_sq1 = nearest_linesegment_point(origin,tip,tip2)
        N2, d_sq2 = nearest_linesegment_point(origin2,tip2,tip)

        # Compensate for radius of probe head
        d_sq1 = d_sq1 - rh2**2
        d_sq2 = d_sq2 - rh**2

        # Also check distance between the tips including their radius
        d_sq3 = (tip2.x-tip.x)**2 + (tip2.y-tip.y)**2 - (rh+rh2)**2

        if (d_sq1 < d_sq2) and (d_sq1 < d_sq3):
            # From middle of this probe to tip of other probe
            A = N1
            dx = tip2.x-A.x
            dy = tip2.y-A.y
            h = np.sqrt(dx**2 + dy**2)
            cos_theta = dx/h
            sin_theta = dy/h
            B = Point(tip2.x-rh2*cos_theta, tip2.y-rh2*sin_theta)
            d_sq = d_sq1
        elif (d_sq2 < d_sq1) and (d_sq2 < d_sq3):
            # From tip of this probe to middle of other probe
            B = N2
            dx = B.x-tip.x
            dy = B.y-tip.y
            h = np.sqrt(dx**2 + dy**2)
            cos_theta = dx/h
            sin_theta = dy/h
            A = Point(tip.x+rh*cos_theta, tip.y+rh*sin_theta)
            d_sq = d_sq2
        else:
            # The tips are closest point of contact
            dx = tip2.x-tip.x
            dy = tip2.y-tip.y
            h = np.sqrt(dx**2 + dy**2)
            if h != 0:
                cos_theta = dx/h
                sin_theta = dy/h
                A = Point(tip.x+rh*cos_theta, tip.y+rh*sin_theta)
                B = Point(tip2.x-rh2*cos_theta, tip2.y-rh2*sin_theta)
                d_sq = d_sq3
            else:
                d_sq = 0
                A = Point(tip.x,tip.y)
                B = Point(tip.x,tip.y)

        if info:
            print d_sq1, d_sq2, d_sq3

        if d_sq < 0:
            d_sq = 0

        return d_sq, A, B

    def collision(self, probe2):
        """ Determine whether this probe has collided with probe2 """

        # Determine cartesian coordinates of line segments
        A = Point(self.x0, self.y0)
        B = Point(self.x, self.y)

        C = Point(probe2.x0, probe2.y0)
        D = Point(probe2.x, probe2.y)

        #print "line dist:", np.sqrt(distsq_lines(A,B,C,D))

        # Check for intersection of line segments:
        lines_intersect = ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        # Check the distance between the two probes
        d,A,B = self.dist(probe2)
        heads_collided = d == 0

        return lines_intersect or heads_collided

    def update_graphics(self):
        """ Update the line graphic to the current probe position """
        self.line.set_data([self.x0, self.x], [self.y0, self.y])
        self.head.center = [self.x,self.y]
        self.trail.set_data(self.trail_x, self.trail_y)

    def pol2cart(self, r, theta):
        """ Covert polar coordinates for this probe into cartesian """
        x = self.x0 + r*np.cos(theta)
        y = self.y0 + r*np.sin(theta)
        return x, y

    def cart2pol(self, x, y):
        """ Convert cartesian to polar coordinates for this probe """
        r = np.sqrt((x-self.x0)**2 + (y-self.y0)**2)
        theta = np.arctan2(y-self.y0,x-self.x0)
        return r, theta

    def check_ranges(self):
        """ Hard stops at maximum rotator offset, extension """

        out_of_range = False
        r = self.r
        theta = self.theta

        thetas = np.unwrap([self.theta_home, self.theta])
        d_theta = thetas[1] - thetas[0]
        errstr = ''
        if abs(d_theta) > theta_max:
            errstr = 'range theta %f (%f)' % (np.degrees(d_theta),theta_max_deg)
            self.theta = self.theta_home + np.sign(d_theta)*theta_max
            self.vt = 0
            out_of_range = True

        if self.r > r_max:
            errstr = errstr + ' range r %f (%f)' % (self.r, r_max)
            self.r = r_max
            self.vr = 0
            out_of_range = True

        if self.r < r_min:
            errstr = errstr + ' range r %f (%f)' % (self.r, r_min)
            self.r = r_min
            self.vr = 0
            out_of_range = True

        if out_of_range:
            self.x, self.y = self.pol2cart(self.r, self.theta)
            raise ProbeLimits(errstr)

    def move(self):
        """ Perform one step of the motion integration """

        if self.moving is None:
            return

        if self.star is None:
            sTarg = Star(self.x_home,self.y_home)
        else:
            sTarg = self.star

        #if use_vel_servo:
        #    self.vr = self.vr + self.ar*dt
        #    self.vt = self.vt + self.at*dt

        #print "move1:",(self.r, np.degrees(self.theta)), \
        #    (self.vr*dt, np.degrees(self.vt*dt))

        s_r,s_t = self.cart2pol(sTarg.x,sTarg.y)
        targ_vr = (s_r - self.r)/dt
        targ_vt = (s_t - self.theta)/dt

        dx = (s_r - self.r)
        dy = (s_t - self.theta)

        if (targ_vr < self.vr) and (targ_vt < self.vt) and False:
        #    print "Here",self
            self.vr = targ_vr
            self.vt = targ_vt
            #r = s_r
            #theta = s_t
        #else:
        r = self.r + self.vr*dt
        theta = self.theta + self.vt*dt
        
        self.set_pol(r, theta)

        #print "move2:",(self.r, np.degrees(self.theta))

    # Potentials from
    #https://www.google.ca/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0CBwQFjAAahUKEwib2cCGiebIAhWDWT4KHe6oAlw&url=http%3A%2F%2Fwww.ro.feri.uni-mb.si%2Fpredmeti%2Frobotizacija%2Fknjiga%2Fplaniranje.pdf&usg=AFQjCNHweou3NCNX3k3_me6DbRBYJ-vqxg&bvm=bv.106130839,d.cWw&cad=rja
    # "Chapter 7: Path Planning and Collision Avoidance"

    def u_target(self, x0, y0):
        """ Evaluate the potential+gradient for a target at x0, y0 """

        d_sq = (self.x-x0)**2 + (self.y-y0)**2
        d = np.sqrt(d_sq)
        if d < tol_attract:
            # Use quadratic potential when near
            u = 0.5*beta*d_sq
            du_dx = -beta*(self.x-x0)
            du_dy = -beta*(self.y-y0)
        else:
            # Conic potential when distant
            u = tol_attract*beta*d - 0.5*beta*tol_attract**2
            du_dx = -tol_attract*beta*(self.x-x0)/d
            du_dy = -tol_attract*beta*(self.y-y0)/d

        grad = np.array([du_dx,du_dy])

        return u, grad

    def u_collision(self, probe2):
        """ Evaluate the potential,gradient,nearpoints for probe2 interaction"""

        # Are probes collided?
        if self.collision(probe2):
            u = u_col_max
            grad = None
            A = None
            B = None
            raise ProbeCollision("Probes collided! (%f,%f) (%f,%f)" % \
                                 (self.x,self.y,probe2.x,probe2.y))
        else:
            # Distance to other probe with clipping
            d_sq,A,B = self.dist(probe2)
            d = np.sqrt(d_sq)

            # Potential
            if d < col_min:
                # We're very close, clipped potential
                u = u_col_max
            else:
                if d < tol_avoid:
                    u = 0.5*alpha*(1/d - 1/tol_avoid)**2
                else:
                    u = 0

            # Direction (downhill points towards this probe from probe2)
            vect = np.array([A.x-B.x,A.y-B.y])
            norm = np.linalg.norm(vect)
            if norm <= 0:
                raise ProbeCollision("Invalid norm detected")
            dir = vect/norm

            # Gradient
            if d < tol_avoid:
                grad_mag = alpha*(1/d - 1/tol_avoid)*1/d**2
                grad = grad_mag*dir
            else:
                du_dx = 0
                du_dy = 0
                grad = np.array([du_dx,du_dy])

        return u, grad, A, B

    def u_ifu(self):
        """ Evaluate the potential,gradient for probe WRT IFU pickoff region"""

        # distance of probe from centre of FOV
        r_sq = self.x**2 + self.y**2
        r = np.sqrt(r_sq)

        # The distance from the probe to the pickoff region needs
        # to account for the size of the prob head and the size of the
        # region
        d = r - (r_head + r_ifu)
        if d <= 0:
            raise ProbeVignetteIFU("Probe vignettes IFU! (%f,%f)" % \
                                    (self.x,self.y))
        else:
            # Use the same collision potential function
            if d < col_min:
                # We're very close, clipped potential
                u = u_col_max
            else:
                if d < tol_avoid:
                    u = 0.5*alpha*(1/d - 1/tol_avoid)**2
                else:
                    u = 0

            # Direction (points towards probe from centre)
            vect = np.array([self.x,self.y])
            norm = np.linalg.norm(vect)
            if norm <= 0:
                raise ProbeVignetteIFU("Invalid norm detected")
            dir = vect/norm

            # Gradient
            if d < tol_avoid:
                grad_mag = alpha*(1/d - 1/tol_avoid)*1/d**2
                grad = grad_mag*dir
            else:
                du_dx = 0
                du_dy = 0
                grad = np.array([du_dx,du_dy])

        return u, grad

    def grad_pol(self, grad):
        """ Convert cartesian gradient (force) in polar given point"""

        # Calculate unit vectors for the radian and theta directions in
        # the cartesian plane
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        r_hat     = np.array([ cos_theta,sin_theta])
        theta_hat = np.array([-sin_theta,cos_theta])

        # Project gradient along these unit vectors.
        # Divide vector in theta_hat direction by r to get
        # an angular offset.

        grad_r = np.dot(grad,r_hat)
        grad_theta = np.dot(grad,theta_hat)/self.r

        return np.array([grad_r,grad_theta])

# --- class State -------------------------------------------------------------

# state of the full set of OIWFS probe arms

class State(object):
    def __init__(self, probes,
                 figsize=(5.5,5),
                 dpi=100,
                 fname=None,
                 head_fill=False,
                 head_lw=2,
                 probe_width=2,
                 title_str=None,
                 aster=None,
                 aster_select=True,
                 use_tran=True,
                 use_vel_servo=False,
                 tran_scale=3,
                 aster_start=0,
                 catalog=None,
                 catalog_start=None,
                 probe_cols=['b','b','b'],
                 display=True,
                 dwell=200,
                 frameskip=1,
                 plotlim=None,
                 contours=None,
                 contour_steps=1,
                 i_ref=None,
                 levels=None,
                 vectors=None,
                 star_vel=None,
                 wcs=None):

        if probes is None:
            # Set up probes around edge of patrol area pointing in, with origins
            # at r_origin from the centre of the FOV
            self.probes=[]
            for phi in np.arange(90, 360.+90, 360./3):
                x = r_origin*np.cos(np.radians(phi))
                y = r_origin*np.sin(np.radians(phi))
                theta = np.radians(np.fmod(phi+180, 360))
                self.probes.append(Probe(x,y,r0,theta))
        else :
            # Probes provided by caller
            self.probes = probes

        self.stars = []       # star objects
        for i in range(maxstars):
            self.stars.append(Star(None,None))

        self.dwell = dwell
        self.dwell_count = dwell  # reset each time we start a move
        self.display = display
        self.move_time = 0
        self.figsize=figsize
        self.dpi=dpi
        self.fname=fname
        self.head_fill=head_fill
        self.head_lw=head_lw
        self.probe_width=probe_width
        self.title_str=title_str
        self.probe_cols=probe_cols
        self.plotlim=plotlim
        self.plot_init()
        self.asterism_counter=0

        self.i_aster = aster_start  # predefined asterism counter
        self.i_vaster = [aster_start,aster_start,aster_start]

        self.aster=aster
        self.aster_select=aster_select

        self.use_tran=use_tran
        self.use_vel_servo=False
        self.tran_scale=tran_scale
        self.frameskip=frameskip

        self.contours=contours
        self.contour_steps=contour_steps
        self.levels=levels
        self.vectors=vectors
        self.vectors_object=None
        self.star_vel=star_vel
        if catalog is not None:
            # File is assumed to consist of x, y columns of positions
            # in degrees. Also assume that they are at a low Dec. So
            # just convert directly into mm offsets using platescale
            self.catalog = catalog
            self.catalog_xdeg, self.catalog_ydeg = np.loadtxt(catalog,unpack=True)
            self.catalog_x = self.catalog_xdeg*3600*platescale  # mm
            self.catalog_y = self.catalog_ydeg*3600*platescale  # mm
            self.catalog_stars = np.array([Star(self.catalog_x[i],self.catalog_y[i]) for i in range(len(self.catalog_x))])
            for i in range(len(self.catalog_stars)):
                self.catalog_stars[i].index = i # hack
            self.catalog_assigned = np.array([False]*len(self.catalog_x))

        # Starting OIWFS pointing provided in catalog coordinates
        if catalog_start:
            self.oiwfs_x0 = catalog_start[0]*3600*platescale # mm
            self.oiwfs_y0 = catalog_start[1]*3600*platescale # mm
        else:
            self.oiwfs_x0 = 0
            self.oiwfs_y0 = 0


        if i_ref is not None:
            self.i_ref = i_ref
            self.p_ref = self.probes[self.i_ref]            # probe of interest
            p_o = set(self.probes).difference([self.p_ref]) # other probes
            # hacky way to get a list
            temp = [p for p in p_o]
            self.p_o1 = temp[0]
            self.p_o2 = temp[1]
        else:
            self.i_ref = None
            self.p_ref = None
            self.p_o1 = None
            self.p_o2 = None

    def plot_init(self):
        """ Initialize the OIWFS plot """

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        all_x0 = np.array([-r_origin,r_origin])
        all_y0 = np.array([-r_origin,r_origin])

        self.ax = self.fig.add_subplot(111, autoscale_on=False,
                                       xlim=all_x0,
                                       ylim=all_y0,
                                       aspect='equal')

        # patrol area
        circle=plt.Circle((0,0),r_patrol,color='k',fill=False)
        self.fig.gca().add_artist(circle)

        # IFU pickoff
        circle_ifu=plt.Circle((0,0),r_ifu,color='g',fill=False)
        self.fig.gca().add_artist(circle_ifu)

        # arcs showing patrol regions for each probe
        theta_region = np.linspace(-theta_max,theta_max,num=50,endpoint=True)
        r_region = np.ones(len(theta_region))*r_max

        for p in self.probes:
            arc_x,arc_y = p.pol2cart(r_region,p.theta_home+theta_region)
            self.ax.plot(arc_x, arc_y, '--', color='gray')

        # axes and titles
        self.ax.grid()

        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')

        # arcmin axes not quite working
        #self.ax2 = self.ax.twinx()
        #plt.axis(np.append(all_x0,all_y0/platescale))
        #self.ax2.set_ylabel('(arcmin)')

        #self.ax2 = self.ax.twiny()
        #plt.axis(np.append(all_x0/platescale,all_y0))
        #self.ax2.set_xlabel('(arcmin)')

        if self.title_str is not None:
            plt.title(self.title_str)

        # all animated elements are initialized here
        for i in range(len(self.probes)):
            p = self.probes[i]
            c = self.probe_cols[i]
            p.line, = self.ax.plot([], [], c, linewidth=self.probe_width)
            p.trail, = self.ax.plot([], [], 'g.', markersize=1)
            p.head = plt.Circle([1000,1000],p.r_head,color=c,zorder=100,
                                fill=self.head_fill,linewidth=self.head_lw)
            self.ax.add_patch(p.head)

        for s in self.stars:
            s.symbol, = self.ax.plot([], [], 'r*', markersize=10,zorder=101)

        self.text = self.ax.text(0.8,0.9,'',transform=self.ax.transAxes)

        if self.plotlim is not None:
            plt.axis(self.plotlim)

        self.graphics_objects = [p.line for p in self.probes]
        self.graphics_objects.extend([p.trail for p in self.probes])
        self.graphics_objects.extend([p.head for p in self.probes])
        self.graphics_objects.extend([s.symbol for s in self.stars])
        self.graphics_objects.append(self.text)

    def random_stars(self):
        """ Generate new random star positions """

        if self.aster:
            #print 'aster: ',self.i_aster

            for i in range(len(self.stars)):
                star = self.stars[i]
                star.x, star.y = self.aster[self.i_aster][i]

            self.i_aster = (self.i_aster+1) % len(self.aster)
            #if self.i_aster > (len(aster)-1):
            #    self.i_aster = len(aster)-1

        else:

            # Select stars uniformly in x,y (and reject outside FOV)
            # to ensure uniform sky density. If I just picked uniform
            # random numbers in theta and r we would have a higher
            # density at the centre of the field. Also reject if too
            # close to other stars.

            for i in range(len(self.stars)):
                star = self.stars[i]
                good = False
                while not good:
                    pos = (np.random.rand(2)*2 - 1)*r_patrol
                    if np.sum(pos**2) <= r_patrol**2:
                        far = True
                        if i >= 1:
                            for j in range(0,i):
                                pos2 = np.array([self.stars[j].x,
                                                 self.stars[j].y])
                                if np.sum((pos2-pos)**2) < r_star**2:
                                    far = False
                                    break
                        if far:
                            good = True
                            star.x, star.y = pos


    def select_probes(self,probe_subset=None,star_vel=None,catalog_subset=None):
        # Select a probe for each star:
        #  - test all probe / star permuations
        #  - reject invalid configurations
        #  - of valid configurations, choose the best
        #  - optionally only reconfigure a subset of the probes
        #  - if star_vel provided, weight configs coming from that direction
        #  - if catalog_subset provided, find best matches from catalog (> 3 stars)
        #
        # It is up to the caller to ensure that catalog_subset only contains
        # stars not currently being tracked (i.e., ensure consistency with
        # probe_subset)

        # Using predefined asterisms / configurations
        if self.aster and self.aster_select:
            for i in range(len(self.stars)):
                p = self.probes[i]
                s = self.stars[i]
                p.star = s
            return True
        
        probes_tracking = [] # indices of tracking probes
        test_star_slots = []
        if probe_subset:
            # If only a subset of the probes are to be reconfigured, figure out
            # which ones are still tracking
            for i in range(len(self.probes)):
                if i not in probe_subset:
                    probes_tracking.append(i)

            # Which star slots are pointed to by the probes to be reconfigured?
            for i in probe_subset:
                for j in range(len(self.stars)):
                    if self.probes[i].star == self.stars[j]:
                        test_star_slots.append(j)

            # Are there any remaining star slots not pointed to by any of the
            # probes?
            for i in range(len(self.stars)):
                s = self.stars[i]
                if s not in [p.star for p in self.probes]:
                    test_star_slots.append(i)

        else:
            # All probes to be reconfigured
            probe_subset=[0,1,2]
            # All star slots are free
            test_star_slots=[0,1,2]

        if catalog_subset is None:
            all_stars = self.stars
        else:
            # Only stars that aren't currently assigned should be in this
            # catalog (i.e., it should be consistent with probe_subset)
            all_stars = self.catalog_stars[catalog_subset]
            for i in range(len(catalog_subset)):
                # Calculate rel position for current OIWFS pointing
                all_stars[i].xrel = all_stars[i].x - self.oiwfs_x0
                all_stars[i].yrel = all_stars[i].y - self.oiwfs_y0
                # Include the catalog index
                all_stars[i].catindex = catalog_subset[i]

        # Remember old probe positions and stars
        old_probe_xy = [(p.x,p.y) for p in self.probes]
        old_star_xy = [(s.x,s.y) for s in self.stars]

        # Set all probes to their star locations -- this is to facilitate
        # configuration testing when only a subset are to be reconfigured
        # since we care about the target config, not where probes are
        # currently
        for p in self.probes:
            if p.star:
                p.set_cart(p.star.x,p.star.y)


        configs = [] # list of valid configs, will contain merit for ranking

        # Iterate over all test star subsets in all_stars

        if len(all_stars) < len(test_star_slots):
            # If we don't have more stars than probes, we look
            # at all the combinations. So, we add in some None stars
            # so that we have at least enough for the number of probes,
            # and any probe assigned None will have to be parked.
            nExtra = len(test_star_slots)-len(all_stars)
            all_stars = np.append(all_stars,[None]*nExtra)

        for test_stars in itertools.combinations(all_stars,len(test_star_slots)):

            # Set star positions to test values
            #for i in range(len(self.stars)):
            for i in range(len(test_stars)):
                test_star_slot = test_star_slots[i]
                if test_stars[i] is not None:
                    self.stars[test_star_slot].x = test_stars[i].xrel
                    self.stars[test_star_slot].y = test_stars[i].yrel
                else:
                    # No star for this slot
                    self.stars[test_star_slot].x = None
                    self.stars[test_star_slot].y = None
                    
            # Check all possible configurations for probes being configured
            for probe_index in itertools.permutations(probe_subset,len(probe_subset)):
                #print 'config:', probe_index

                probe_limits = []
                probe_collisions = []
                probe_vignette = []
                
                # Test probes in this configuration
                for i in range(len(probe_subset)):
                    try:
                        test_star_slot = test_star_slots[i]
                        s = self.stars[test_star_slot]
                        p = self.probes[probe_index[i]]
                        if s.x is not None:
                            p.set_cart(s.x,s.y)
                            p.u_ifu()
                            p.star = s
                        else:
                            # Park this probe
                            p.set_cart(p.x_home,p.y_home)
                            p.star = None
                    except ProbeLimits:
                        # Configuration exceeds probe actuator limits
                        #print "Exceed probe limits."
                        #continue
                        probe_limits.append(i) # index into probe_subset
                    except ProbeVignetteIFU:
                        probe_vignette.append(i)

                # Check for probe collisions (using minimum star distance as
                # threshold) in this configuration. Note that we check all
                # probes here, not just the subset being reconfigured.
                #collides = False
                for i in itertools.combinations(range(3),2):
                    #print "check config: (%i,%i)" % (i[0], i[1])
                    this_d,a,b = self.probes[i[0]].dist(self.probes[i[1]])
                    #print '     this_d=',this_d
                    if this_d < r_star:
                        probe_collisions.append([i[0],i[1]]) # absolute probe indices
                        #print '  bad'
                        #collides = True
                        #break
                #if collides:
                    #print "collides"
                #    continue
                #print '  good'

                # Calculate figure of merit.
                d = None # used to calculate merit. Array of distance units for 
                         # probes to be reconfigured
                if self.star_vel:
                    # For non-sidereal tracking we want to choose
                    # stars that are closer to the direction from which
                    # they are moving into the field of view.
                    #
                    # We first calculate the equation for a line that
                    # is tangent to the FOV circle on the side from
                    # which the stars are coming, perpendicular to the
                    # apparent velocity vector of those stars. We then calculate
                    # the distance of the stars to that line.
                    #
                    # In the event that a probe exceeds its limits,
                    # the merit is assigned a poor value, and
                    # the probe in question is later parked.
                    #
                    # In the event of a collision, the merit is also
                    # assigned a poor value. For the two probes involved in the collision,
                    # the lower merit probe is parked.
                    
                    vx = self.star_vel[0]
                    vy = self.star_vel[1]

                    if vy == 0:
                        # Stars are moving horizontally
                        #merit = np.max(np.abs(np.array([-np.sign(vx)*r_patrol-self.probes[i].x for i in probe_subset])))
                        d = np.abs(np.array([-np.sign(vx)*r_patrol-self.probes[i].x for i in probe_subset]))
                    elif vx == 0:
                        # Stars are moving vertically
                        #merit = np.max(np.abs(np.array([-np.sign(vy)*r_patrol-self.probes[i].y for i in probe_subset])))
                        d = np.abs(np.array([-np.sign(vy)*r_patrol-self.probes[i].y for i in probe_subset]))
                    else:
                        # Full solution. First solve for x, y values of the tangent
                        # point to the circle along the line representing the
                        # star velocity vector that goes through the origin
                        xtan = -np.sign(vx)*np.sqrt(r_patrol**2 / ((vy/vx)**2+1))
                        ytan = -np.sign(vy)*np.sqrt(r_patrol**2 / ((vx/vy)**2+1))

                        # The tangent line has a slope orthogonal to velocity vector
                        m = -vx/vy
                        b = ytan - m*xtan

                        #merit = np.max(np.array([dist_line_point(m,b,self.probes[i]) for i in probe_subset]))
                        d = np.array(np.abs([dist_line_point(m,b,self.probes[i]) for i in probe_subset]))

                    # Fix up d for any probe that is parked
                    # Haven't tested yet
                    #for i in range(len(probe_subset)):
                    #    p = self.probes[probe_subset[i]]
                    #    if p.star is None:
                    #        d[i] = 0

                    # set d to a large number if newly-configured probe in limit
                    for i in probe_limits:
                        d[i] = d_limit

                    # set d to a large number of probe vignettes IFU pickoff
                    for i in probe_vignette:
                        d[i] = d_collided # just reuse value

                    # set d of worst probe involved in collision to large number

                    # first we need a mapping from absolute probe number to
                    # probe index within the subset
                    subset_map = [None]*3
                    for i in range(len(probe_subset)):
                        subset_map[probe_subset[i]] = i

                    for c in probe_collisions:
                        # Check if the first probe is in the reconfig subset
                        if c[0] in probe_subset:
                            # Check if the other probe is in the reconfig subset
                            if c[1] in probe_subset:
                                # Set the one with the larger d to a large number
                                if d[subset_map[c[0]]] > d[subset_map[c[1]]]:
                                    d[subset_map[c[0]]] = d_collided
                                else:
                                    d[subset_map[c[1]]] = d_collided
                            else:
                                # It isn't, so first probe large number
                                d[subset_map[c[0]]] = d_collided 
                        else:
                            # It isn't so check if the second probe to be reconfiged
                            if c[1] in probe_subset:
                                # second probe large number
                                d[subset_map[c[1]]] = d_collided
                                

                    # Merit is the sum of d (i.e., a bigger number is worse)
                    merit = np.sum(d)

                else:
                    # otherwise we prefer configurations that minimize
                    # the maximum probe extension
                    d = np.array([self.probes[i].r for i in probe_subset])
                    merit = np.max(d)

                configs.append({
                    'stars':test_stars,
                    'probes':probe_index,
                    'd':d,
                    'merit':merit})

        # Return probes and stars to old settings
        for i in range(len(self.probes)):
            self.probes[i].set_cart(old_probe_xy[i][0], old_probe_xy[i][1])
        for i in range(len(self.stars)):
            self.stars[i].x = old_star_xy[i][0]
            self.stars[i].y = old_star_xy[i][1]

        # Select the best configuration
        min_merit = None
        best = None

        for c in configs:
            # Loop over possible probe configurations
            #print c
            if best is None:
                best = c['probes']
                min_merit = c['merit']
                best_stars = c['stars']
                best_d = c['d']
            elif c['merit'] < min_merit:
                min_merit = c['merit']
                best = c['probes']
                best_stars = c['stars']
                best_d = c['d']
    
        if best is not None:            
            # Check the d array to see if the probe is to be parked
            # because it would be in a limit, or collide with another probe
            for i in range(len(probe_subset)):
                p = self.probes[best[i]] #[probe_subset[i]]
                if (best_d[i] == d_limit) or (best_d[i] == d_collided):
                    # Park because limit or collided
                    p.park = True
                    if p.star is not None:
                        p.star.index = False
                        p.star = None
                else:
                    # Assigned to star
                    p.park = False

            # Now assign the best star coordinates to the appropriate slot.
            for i in range(len(test_stars)):
                p = self.probes[best[i]]
                if p.park == False:
                    # We may have flagged it to park above

                    test_star_slot = test_star_slots[i]
                    s = self.stars[test_star_slot]
                    
                    if best_stars[i] is None:
                        # this probe will actually be parked
                        p.park = True
                        p.star = None
                    else:
                        # Update star positions to best values
                        s.x = best_stars[i].xrel
                        s.y = best_stars[i].yrel

                        # Record the catalog index of the star
                        s.catindex = best_stars[i].catindex

                        # Indicate that this catalog star is assigned
                        self.catalog_assigned[s.catindex] = True

                        # Update probes to include selected star references
                        p.star = s
        else:
            # Couldn't find a star + probe assignment configuration. So,
            # probes that were to be reconfigured are assigned None star
            for i in probe_subset:
                p = self.probes[i]
                p.park = True
                if p.star is not None:
                    p.star.index = False
                    p.star = None
            return False
 
    
        # update which catalog stars assigned
        #if catalog_subset is not None:
        #    for s in best_stars:
        #        self.catalog_assigned[s.index] = True

        for i in range(3):
            p = self.probes[i]
            if p.star is None:
                sTarg = Star(p.x_home,p.y_home)
            else:
                sTarg = p.star

            #print 'Select Probe',i,':',p.x,p.y,sTarg.x,sTarg.y

        return True

    def set_selected(self):
        """ Set probe arm positions to selected star positions """
        for p in self.probes:
            s = p.star
            p.set_cart(s.x, s.y)



    def move_probes(self,justGradient=False,info=False):
        """ Move selected probes toward target through potential field"""

        # Remember old probe positions/velocities
        old_state = [(p.r,p.theta,p.vr,p.vt) for p in self.probes]

        # Each probe has an attractive potential toward its' target,
        # and a repulsive potential from other probes to avoid
        # collisions. We will sum the gradients from all of these
        # contributions to figure out the direction of the next step
        probe_gradients = [ {'grad_targ': np.array([0,0]),
                             'grad_col' : np.array([0,0]),
                             'grad_tran': np.array([0,0])} \
                            for i in range(3) ]

        # First do the repulsion - all unique combinations of probes.
        for i,j in itertools.combinations(range(3),2):
            probe_i = self.probes[i]
            probe_j = self.probes[j]

            # Calculate the potential gradient for probe_i
            u,grad_i,P_i,P_j = probe_i.u_collision(probe_j)

            # Make the forces symmetric: probe_j experiences an equal
            # but opposite force to that of probe i.

            grad_j = -grad_i
            grads = [grad_i,grad_j]

            # Do the transverse component if requested
            if self.use_tran:
                # Need to order both ways due to different goal positions
                for a,b in [(i,j),(j,i)]:
                    probe_a = self.probes[a]
                    probe_b = self.probes[b]

                    if a == i:
                        grad_a = grads[0]
                    else:
                        grad_a = grads[1];

                    # Figure out if, in moving towards its goal in a
                    # straight line, probe_a would intersect
                    # probe_b. If so, add a transverse component to
                    # the gradient, orthogonal to the repulsive force,
                    # choosing the direction parallel to the other probe (i.e.,
                    # such that it would cause probe_a to sweep around
                    # probe_b towards the centre of the FOV).

                    # We use a test probe simply to check whether a
                    # straight-line trajectory from the current probe
                    # position to the target intersects the other
                    # probe, so we set the probe head size to
                    # 0. Origin at tip of probe_a, terminates at
                    # target star.

                    if probe_a.star:
                        aTarg = probe_a.star
                    else:
                        aTarg = Star(probe_a.x_home,probe_a.y_home)

                    if probe_b.star:
                        bTarg = probe_b.star
                    else:
                        bTarg = Star(probe_b.x_home,probe_b.y_home)

                    pt = Probe(probe_a.x, probe_a.y, 0, 0, r_head=0)
                    (pt.x, pt.y) = (aTarg.x, aTarg.y)

                    if info:
                        print "here: ",pt.x0,pt.y0,pt.x,pt.y,probe_b.x,probe_b.y

                    if pt.collision(probe_b):
                        # Is this probe's target star at a
                        # counter-clockwise (positive) or clockwise
                        # (negative) relative rotation about the
                        # interactng probe's origin with respect to
                        # the _target_ location of its tip? This lets us choose
                        # the correct vector orthogonal to the
                        # repulsive vector.

                        s_r,s_theta = probe_b.cart2pol(aTarg.x,
                                                       aTarg.y)

                        t_r,t_theta = probe_b.cart2pol(bTarg.x,
                                                       bTarg.y)

                        thetas = np.unwrap([s_theta,t_theta])


                        if thetas[0] > thetas[1]:
                            # counter-clockwise
                            grad_tran_a = np.array([-grad_a[1],grad_a[0]])
                        else:
                            # clockwise
                            grad_tran_a = np.array([grad_a[1],-grad_a[0]])

                        norm = np.linalg.norm(grad_tran_a)
                        if norm < 0:
                            raise ProbeCollision("Invalid norm target vector.")
                        elif norm == 0:
                            grad_tran_a = grad_tran_a * 0
                        else:
                            grad_tran_a = grad_tran_a/norm

                        norm = self.tran_scale*np.linalg.norm(grad_a)
                        if norm < 0:
                            raise ProbeCollision("Invalid norm grad vector")

                        if info:
                            print a, b, np.degrees(thetas), \
                                grad_a, grad_tran_a

                        # Same magnitude as repulsive potential
                        grad_tran_a = norm*grad_tran_a
                        grad_tran_b = -grad_tran_a


                        probe_gradients[a]['grad_tran'] = probe_gradients[a]['grad_tran'] + grad_tran_a
                        probe_gradients[b]['grad_tran'] = probe_gradients[b]['grad_tran'] + grad_tran_b


            probe_gradients[i]['grad_col'] = probe_gradients[i]['grad_col'] + \
                                             grad_i
            probe_gradients[j]['grad_col'] = probe_gradients[j]['grad_col'] + \
                                             grad_j

        # Repulsion due to IFU pickoff
        for i in range(3):
            probe = self.probes[i]
            u,grad = probe.u_ifu()
            probe_gradients[i]['grad_ifu'] = grad

        # Now do the attractive potentials
        # If a probe isn't currently assigned to a star, point it toward its
        # home position
        for i in range(3):
            probe = self.probes[i]

            if probe.star:
                sTarg = probe.star
            else:
                sTarg = Star(probe.x_home,probe.y_home)

            u,grad = probe.u_target(sTarg.x,sTarg.y)

            #print "Attract",i,':',u,grad

            probe_gradients[i]['grad_targ'] = grad

        if justGradient:
            # If we just want the gradient calculation, return here
            # and don't move anything.
            return probe_gradients

        # Add gradients together, convert to polar, figure out where
        # they point, and update the probe velocities
        for i in range(3):
            p = self.probes[i]
            if p.star:
                s = p.star
            else:
                s = Star(p.x_home,p.y_home)

            theta = p.theta
            r = p.r

            dist_sq = (p.x-s.x)**2 + (p.y-s.y)**2
            if dist_sq < tol_sq:
                # Are we really close to the target?
                p.vr = 0
                p.vt = 0

                if self.star_vel:
                    # If tracking moving stars, set probe positions to
                    # the target and continue to next probe to make
                    # animation smoother

                    old_x = p.x
                    old_y = p.y
                    try:   
                        p.set_cart(s.x,s.y)
                    except (ProbeLimits,ProbeCollision,ProbeVignetteIFU):
                        # Can't go here so just revert the move for now. The invalid
                        # target star will be caught elsewhere and trigger a reconfig
                        p.set_cart(old_x,old_y)
                    p.trail_x.append(p.x)
                    p.trail_y.append(p.y)
                
                else:
                    # We're there, so stop moving
                    p.moving = None
                continue
                #else:
                    #print 'Overshot! (%f,%f) > (%f,%f)' % \
                    #    (p.vr, p.vt, vr_tol, vt_tol)

            if dist_sq < (2*tol)**2:
                # Additional hack: if we're close enough to the
                # target, ignore the repulsive components and head
                # straight there. This is kind of like following TCS
                # demands when in closed-loop.  This should avoid a
                # probe that is *not* tracking from "moving" a probe
                # that is tracking when it passes by
                grad_cart = probe_gradients[i]['grad_targ']
            else:
                # Otherwise figure out velocities from gradient
                grad_cart = probe_gradients[i]['grad_col'] + \
                            probe_gradients[i]['grad_tran'] + \
                            probe_gradients[i]['grad_ifu'] + \
                            probe_gradients[i]['grad_targ']

            grad = p.grad_pol(grad_cart)
            norm = np.linalg.norm(grad)
            #if i == 0:
            #    print 'Total polar gradients',i,':',grad, norm, [p.x, p.y], \
            #        [p.r,p.theta]
            #time.sleep(0.1)
                #print norm
            if norm > 0 :

                #print i, norm, grad_tol
                #if (len(p.trail_x) > 5):
                #    d = np.linalg.norm(np.array([p.trail_x[-1]-p.trail_x[-3], \
                #                                 p.trail_y[-1]-p.trail_y[-3]]))
                    #print i, d, tol_stuck
                #else:
                #    d = None

                #if d is not None and d < tol_stuck:

                l = 30
                if len(p.trail_x) > l:
                    x_m = np.mean(p.trail_x[-l:])
                    y_m = np.mean(p.trail_y[-l:])
                    x_r = np.max(p.trail_x[-l:]) - np.min(p.trail_x[-l:])
                    y_r = np.max(p.trail_y[-l:]) - np.min(p.trail_y[-l:])
                else:
                    x_m = None

                if x_m is not None and x_r < tol_stuck and y_r < tol_stuck:
                    print "Stuck:", i, x_m, y_m, x_r, y_r, tol_stuck

                #if norm < grad_tol:
                    # If the gradient is small enough, and we haven't
                    # reached the target, we must have fallen in a
                    # local minimum. Locate the nearest probe arm and
                    # switch both to retract mode until they are short
                    # enough to pass eachother.

                    j_nearest = None
                    nearest_dist = 0
                    for j in set(range(3)).difference([i]):
                        p_other = self.probes[j]
                        d_sq = p.dist(p_other)
                        if (j_nearest is None) or (d_sq < nearest_dist):
                            j_nearest = j
                            nearest_dist = d_sq

                    p_nearest = self.probes[j]

                    # Check to make sure that the arms can actually collide
                    # if pointing right at eachother
                    if p.r + p_nearest.r < d_clear:
                        print "Error: Don't know how to handle local minimum."
                        #sys.exit(1)

                dir = grad/norm

                max_vr,max_vt = max_vel(dir)
                    
                if self.use_vel_servo:
                    #info = p is self.probes[1]
                    info = False

                    # Work out the accelerations, new velocities
                    star_r, star_theta = p.cart2pol(s.x, s.y)

                    delta_r = star_r - p.r
                    thetas = np.unwrap([p.theta,star_theta])
                    delta_theta = thetas[1]-thetas[0]
                    p.ar = calc_accel(p.r, p.vr, max_vr, delta_r, ar, info=info)
                    p.at = calc_accel(thetas[0], p.vt, max_vt, delta_theta, at,
                                      info=info)

                    if info:
                        print '    r: pos=%f targ=%f diff=%f accel=%f v=%f vtarg=%f' % \
                            (p.r, star_r, delta_r, p.ar, p.vr, max_vr)

                        print 'theta: pos=%f targ=%f diff=%f accel=%f v=%f vtarg=%f' % \
                            (thetas[0], thetas[1], delta_theta, p.at, p.vt, max_vt)

                    p.vr = p.vr + p.ar*dt
                    p.vt = p.vt + p.at*dt

                    if abs(p.vr) > abs(max_vr):
                        p.vr = np.sign(p.vr)*max_vr
                    if abs(p.vt) > abs(max_vt):
                        p.vt = np.sign(p.vt)*max_vt

                else:
                    # Set velocity to new target
                    p.vr,p.vt = max_vr,max_vt

            # Try moving the probes
            p.moving = 'moving'
            try:
                p.move()
            except ProbeLimits:
                # Asterism should be OK so let actuators hit the rails
                continue

            # Update probe trails
            p.trail_x.append(p.x)
            p.trail_y.append(p.y)

    def init_animation(self):
        for p in self.probes:
            p.line.set_data([], [])
            p.trail.set_data([], [])
            p.head.center = [1000,1000]

        for s in self.stars:
            s.symbol.set_data([], [])

        return self.graphics_objects

    def animate(self, i, single=False):

        if self.fname: # and (self.vectors or self.contours):
            print "Animate frame",i

        objects = copy.copy(self.graphics_objects) # shallow copy

        for frame in range(self.frameskip):
            # For each call to animate, only a single frame will be displayed,
            # but frameskip frames will be calculated
            if frame == 0:
                show=True
            else:
                show=False

            update = True

            if self.star_vel:
                if self.catalog:
                    # We're crab-walking through a catalog                
                    
                    # Move the visible stars. We do this first to ensure
                    # that we're checking stars are at valid positions
                    # before deciding on reconfigs
                    for s in self.stars:
                        if s.x is not None and s.y is not None:
                            s.x = s.x + self.star_vel[0]*dt
                            s.y = s.y + self.star_vel[1]*dt

                    # The OIWFS pointing moves in the opposite sense of star_vel
                    self.oiwfs_x0 = self.oiwfs_x0 - self.star_vel[0]*dt
                    self.oiwfs_y0 = self.oiwfs_y0 - self.star_vel[1]*dt


                    # Which stars are in the OIWFS patrol area
                    star_dist = np.sqrt((self.catalog_x - self.oiwfs_x0)**2 + \
                        (self.catalog_y - self.oiwfs_y0)**2)
                    infield = np.where((star_dist <= r_patrol) & (self.catalog_assigned == False))[0]

                    # Initial assignment of stars
                    if all(s.x is None for s in self.stars):
                        success = self.select_probes(catalog_subset=infield)
                        if not success:
                            print("Could not establish starting config")
                            sys.exit(1)

                    # Probes tracking stars that are no longer valid targets
                    # get flagged as needing reconfiguration.
                    # Also, any probe not currently tracking a star is also
                    # flagged.
                    need_reconfig=[]  # which probes need reconfig
                    for k in range(len(self.probes)):
                        reconfig = False  # if current probe needs reconfig
                        p = self.probes[k]
                        if p.star is None:
                            reconfig = True
                        else:
                            # If the star is no longer within the FOV we
                            # need to reconfig
                            reconfig = False
                            this_star_dist = np.sqrt(p.star.x**2 + p.star.y**2)

                            if this_star_dist > r_patrol:
                                reconfig = True
                            else:
                                # try setting probe to current star position
                                # and check for exceeding probe limits or
                                # collisions, or vignetting the IFU
                                old_x = p.x
                                old_y = p.y

                                try:
                                    p.set_cart(p.star.x,p.star.y)
                                    p.u_ifu()
                                except ProbeLimits:
                                    reconfig = True
                                except ProbeCollision:
                                    reconfig = True
                                except ProbeVignetteIFU:
                                    reconfig = True
                                
                                # revert position after test
                                try:
                                    p.set_cart(old_x,old_y)
                                except ProbeLimits:
                                    # hack
                                    continue

                        if reconfig:
                            if p.star is not None:
                                # can't move here so stop tracking
                                self.catalog_assigned[p.star.catindex] = False
                                p.star = None
                            
                            # Reset trail and flag for reconfig
                            p.trail_x = []
                            p.trail_y = []
                            need_reconfig.append(k)
                            #print("Here",k)

                    # Perform reconfigs
                    if need_reconfig:
                        success = self.select_probes(probe_subset=need_reconfig, \
                            catalog_subset=infield)
                        #if not success:
                        #    print("Could not reconfig probes")
                        #    sys.exit(1)

                    
                    # Hacking to see if starfield moves, so no
                    # probe movement.
                    #update = False
                    
                else:
                    # Moving several canned stars
                    for j in range(len(self.stars)):
                        s = self.stars[j]
                        p = self.probes[j]
                        p.star = s

                        if s.x is None or s.y is None:
                            s.x,s.y = self.aster[0][j]

                        xnew = s.x + self.star_vel[0]*dt
                        ynew = s.y + self.star_vel[1]*dt

                        r,theta=p.cart2pol(xnew,ynew)

                        if (np.sqrt(xnew**2 + ynew**2) < r_patrol) and \
                        (r < r_max):
                            # Move star if within FOV
                            s.x = xnew
                            s.y = ynew
                        else:
                            # Otherwise switch to next star
                            self.i_vaster[j] = (self.i_vaster[j]+1) % \
                                                len(self.aster)
                            s.x,s.y = self.aster[self.i_vaster[j]][j]
                            p.trail_x = []
                            p.trail_y = []

            else:
                # Select new stars if we're stopped
                stopped = True
                for p in self.probes:
                    if p.moving:
                        stopped = False
                    #s = p.star
                    #if s is None:
                    #    continue
                    #dist_sq = (p.x-s.x)**2 + (p.y-s.y)**2
                    #if dist_sq > tol_sq:
                        # Not there yet, so system is not stopped
                    #    stopped = False

                if stopped:
                    if self.dwell_count < self.dwell:
                        self.dwell_count = self.dwell_count + 1
                        if self.display:
                            self.text.set_text('t='+str(self.move_time)+' s')
                        update = False

                        if single:
                            # If only plotting one trajectory, exit this way
                            return None
                    else:
                        for p in self.probes:
                            p.trail_x = []
                            p.trail_y = []
                        success = False
                        while success is False:
                            self.random_stars()
                            try:
                                success = self.select_probes()
                            except:
                                pass
                        self.dwell_count = 0
                        if self.display:
                            for s in self.stars:
                                s.update_symbol()
                        if self.move_time:
                            print '   reconfig time:',self.move_time
                        self.move_time = 0
                        #print 'selected stars:', \
                        #    [(s.x,s.y) for p in self.probes]

            # Hack: always draw stars second time this method is
            # called. Otherwise the first asterism will not be visible
            if ((i == 1) or self.star_vel) and self.display:
                for s in self.stars:
                    s.update_symbol()

            # Move the probes
            if update:
                if self.move_time == 0:
                    # if move_time is 0 we're starting a new move
                    self.asterism_counter = self.asterism_counter+1
                    print '*** asterism %i (%i)***' % (self.asterism_counter,i)
                    print 'current position:', \
                        [(p.x,p.y) for p in self.probes]
                    print '    target stars:', \
                        [(p.star.x,p.star.y) for p in self.probes]

                self.move_time += dt
                self.move_probes()
                for p in self.probes:
                    if show and p.moving and self.display:
                        p.update_graphics()

                        # Contours and vectors change with probe movement
                        if self.contours:
                            self.plot_contours()

                        if self.vectors:
                            if self.vectors_object:
                                self.vectors_object.remove()
                            self.vectors_object = self.plot_vectors()
                            objects.append(self.vectors_object)

        # Return objects involved with animation
        return objects

    def plot_contours(self):
        samples = np.arange(-r_patrol,r_patrol,self.contour_steps)
        x_grid = []
        y_grid = []
        ut_grid = []
        uc_grid = []
        old_x,old_y = [self.p_ref.x,self.p_ref.y]
        for i in range(len(samples)):
            for j in range(len(samples)):
                x = samples[i]
                y = samples[j]

                if x**2 + y**2 > r_patrol**2:
                    continue
                else:
                    self.p_ref.x,self.p_ref.y=(x,y)
                    if self.p_ref.star:
                        sTarg = self.p_ref.star
                    else:
                        sTarg = Star(self.p_ref.x_home,self.p_ref.y_home)

                    ut,gradt = self.p_ref.u_target(sTarg.x,sTarg.y)
                    try:
                        uc_o1,gradc_o1,P_o1,junk = self.p_ref.u_collision(self.p_o1)
                        uc_o2,gradc_o2,P_o2,junk = self.p_ref.u_collision(self.p_o2)
                        uc_o = uc_o1+uc_o2
                    except:
                        uc_o = u_col_max

                    ut_grid.append(ut)
                    uc_grid.append(uc_o)
                    x_grid.append(x)
                    y_grid.append(y)

        self.p_ref.x,self.p_ref.y = [old_x,old_y]

        if self.levels is not None:
            l = levels
        else:
            l = np.logspace(-1.5,0,30,endpoint=True)

        u_grid = np.array(uc_grid)+np.array(ut_grid)

        if 'rep' in self.contours:
            cc = plt.tricontour(x_grid,y_grid,uc_grid,colors='r',levels=l)
        if 'att' in self.contours:
            ct = plt.tricontour(x_grid,y_grid,ut_grid,colors='g',levels=l)
        if 'tot' in self.contours:
            c = plt.tricontour(x_grid,y_grid,u_grid,colors='k',levels=l)


    def plot_vectors(self):
        samples = np.arange(-r_patrol,r_patrol,5)
        x_grid = [] # arrow location
        y_grid = [] # "
        u_grid = [] # arrow x-component
        v_grid = [] # "     y
        c_grid = [] # colour value
        old_x,old_y = [self.p_ref.x,self.p_ref.y]
        for i in range(len(samples)):
            for j in range(len(samples)):
                x = samples[i]
                y = samples[j]

                if x**2 + y**2 > r_patrol**2:
                    continue
                else:
                    self.p_ref.x,self.p_ref.y=(x,y)

                    try:
                        grads = self.move_probes(justGradient=True)
                        grad = grads[self.i_ref]
                        grad_total = np.array([0,0])
                        if 'att' in self.vectors:
                            grad_total = grad_total + grad['grad_targ']
                        if 'rep' in self.vectors:
                            grad_total = grad_total + grad['grad_col']
                        if 'tran' in self.vectors:
                            grad_total = grad_total + grad['grad_tran']

                        norm = np.linalg.norm(grad_total)
                        if norm > 0:
                            grad_total = grad_total / norm
                            x_grid.append(x)
                            y_grid.append(y)
                            u_grid.append(grad_total[0])
                            v_grid.append(grad_total[1])
                            c_grid.append(norm)
                    except:
                        pass

        self.p_ref.x,self.p_ref.y = [old_x,old_y]
        #vectors_object = plt.quiver(x_grid,y_grid,u_grid,v_grid,
        #                            pivot='middle', units='xy',scale=0.2,
        #                            angles='xy', width=0.4,
        #                            facecolor='g',edgecolor='g')


        c_grid = np.log10(np.array(c_grid)/beta)
        #print "norms b4:",np.min(c_grid),np.max(c_grid), \
        #    np.median(c_grid),np.mean(c_grid),np.std(c_grid)

        #c_grid = 1. - np.clip(c_grid-0.7,0.0,0.03)/0.06
        #cmap = cmap=plt.get_cmap('brg')

        c_grid = 0.75 - np.clip(0.325+(c_grid-0.70)*10.,0.0,0.75)
        cmap = cmap=plt.get_cmap('gist_rainbow')

        #c_grid = np.clip(c_grid-0.7,0.05,0.1)/0.15
        #cmap = cmap=plt.get_cmap('YlOrRd')

        #print "norms af:",np.min(c_grid),np.max(c_grid),np.median(c_grid)
        vectors_object = plt.quiver(x_grid,y_grid,u_grid,v_grid,c_grid,
                                    pivot='middle', units='xy',scale=0.2,
                                    angles='xy', width=0.4,
                                    cmap=cmap,
                                    norm=colors.Normalize(vmin=0,vmax=1))



        return vectors_object


# --- replacement Mencoder Writer ---------------------------------------------
# A hack so that we can use an x264-based codec with mencoder


class MencoderBase:
    exec_key = 'animation.mencoder_path'
    args_key = 'animation.mencoder_args'

    # Mencoder only allows certain keys, other ones cause the program
    # to fail.
    allowed_metadata = ['name', 'artist', 'genre', 'subject', 'copyright',
                        'srcform', 'comment']

    # Mencoder mandates using name, but 'title' works better with ffmpeg.
    # If we find it, just put it's value into name
    def _remap_metadata(self):
        if 'title' in self.metadata:
            self.metadata['name'] = self.metadata['title']

    @property
    def output_args(self):
        self._remap_metadata()
        args = ['-o', self.outfile, '-ovc', 'x264']
        if self.bitrate > 0:
            args.append('vbitrate=%d' % self.bitrate)
        if self.extra_args:
            args.extend(self.extra_args)
        if self.metadata:
            args.extend(['-info', ':'.join('%s=%s' % (k, v)
                         for k, v in self.metadata.items()
                         if k in self.allowed_metadata)])
        return args

class MencoderWriter(animation.MovieWriter, MencoderBase):
    def _args(self):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return [self.bin_path(), '-', '-demuxer', 'rawvideo', '-rawvideo',
                ('w=%i:h=%i:' % self.frame_size +
                'fps=%i:format=%s' % (self.fps,
                                      self.frame_format))] + self.output_args

# Combine Mencoder options with temp file-based writing
class MencoderFileWriter(animation.FileMovieWriter, MencoderBase):
    supported_formats = ['png', 'jpeg', 'tga', 'sgi']

    def _args(self):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return [self.bin_path(),
                'mf://%s*.%s' % (self.temp_prefix, self.frame_format),
                '-frames', str(self._frame_counter), '-mf',
                'type=%s:fps=%d' % (self.frame_format,
                                    self.fps)] + self.output_args

# -----------------------------------------------------------------------------

# Generate plots of different probe configurations

def run_sim(animate='cont',             # one of 'cont','track',None
            dwell=200,                  # Dwell time
            contours=None,              # list of ['att','rep','tot']
            levels=None,                # levels for contours
            contour_steps=1,            # spatial resolution in mm
            vectors=None,               # list of ['att','rep','tran']
            i_ref=None,                 # reference probe index (0,1,2)
            startpos=None,              # start positions for probes
            aster=None,                 # sequence of predefined asterisms
            aster_select=False,         # use predefined star selection?
            aster_start=0,              # Start index in aster
            catalog=None,               # Provide filename of star catalog (deg)
            catalog_start=None,         # Starting OIWFS coordinates (deg)
            use_tran=True,              # use transverse component?
            tran_scale=3,               # strength of transverse component
            figsize=(5.5,5),            # figure size
            dpi=100,                    # fig resolution
            fname=None,                 # output to file with this name?
            use_merit=False,            # intelligent star selection
            use_vel_servo=False,        # use velocity servo for motion
            head_fill=True,             # fill probe heads in display?
            head_lw=2,                  # width of the probe head lines
            probe_width=2,              # width of the probe bodies
            plotlim=None,               # limits for plot
            title_str=None,             # plot title
            display=True,               # Display the model
            annotations=None,           # Extra things to plot
            frames=None,                # Number of frames to animate
            frameskip=1,                # Display 1/frameskip frames of ani
            fps=None,                   # fps for animation
            bitrate=3000,               # bitrate if animation to file
            star_vel=None               # move stars across focal plane
):

    # Set the initial OIWFS state and plot

    if i_ref is not None:
        probe_cols = ['gray','gray','gray']
        probe_cols[i_ref]='b'
    else:
        probe_cols = ['b','b','b']

    s = State(None,
              figsize=figsize,
              dpi=dpi,
              fname=fname,
              title_str=title_str,
              probe_width=probe_width,
              head_fill=head_fill,
              head_lw=head_lw,
              aster=aster,
              aster_select=aster_select,
              use_tran=use_tran,
              use_vel_servo=use_vel_servo,
              tran_scale=tran_scale,
              aster_start=aster_start,
              catalog=catalog,
              catalog_start=catalog_start,
              probe_cols=probe_cols,
              display=display,
              dwell=dwell,
              frameskip=frameskip,
              plotlim=plotlim,
              contours=contours,
              contour_steps=contour_steps,
              i_ref=i_ref,
              levels=levels,
              vectors=vectors,
              star_vel=star_vel)

    if animate == 'cont':
        # continuous animation
        if fname is None:
            blit=True
        else:
            blit=False

        ani = animation.FuncAnimation(s.fig, s.animate,
                                      blit=blit,
                                      interval=0.,
                                      frames=frames,
                                      init_func=s.init_animation)

        if fname is not None:
            if fps is None:
                fps = 1/dt
            #Writer = animation.writers['mencoder']
            #writer = Writer(fps=fps,metadata=dict(artist='Me'),bitrate=bitrate)
            #writer = MencoderWriter(fps=fps,metadata=dict(artist='Me'))
            writer = MencoderFileWriter(fps=fps,metadata=dict(artist='Me'))

            ani.save(fname, writer=writer,fps=fps, dpi=dpi)


    elif animate == 'track':
        # plot of a single re-configuration track

        s.init_animation()

        for i in range(3):
            # Setup initial positions
            star = aster[s.i_aster][i]
            p = s.probes[i]
            p.set_cart(star[0],star[1])
            p.update_graphics()

        # Move will be to next asterism in list
        s.i_aster = (s.i_aster+1) % len(aster)

        i=0
        while True:
            result = s.animate(i,single=True)
            if result is None:
                break
            i = i + 1

    else:
        # Display a single staged-configuration

        for j in range(3):
            p = s.probes[j]
            p.star = s.stars[j]
            star1 = aster[s.i_aster][j]    # current position
            p.set_cart(star1[0],star1[1])

            n = (s.i_aster+1) % len(aster)
            star2 = aster[n][j]  # where we're headed
            p.star.x,p.star.y=star2

        if startpos is not None:
            for i_str in startpos:
                i = int(i_str)
                s.probes[i].set_cart(startpos[i_str][0],startpos[i_str][1])

        for p in s.probes:
            p.update_graphics()
            for star in s.stars:
                star.update_symbol()

        # Evaluate and plot potentials as contours.
        if contours:
            s.plot_contours()

        # Evaluate and plot vector field.
        if vectors:
            s.plot_vectors()

        # Plot any extra annotations
        if annotations is not None:
            for item,pars in annotations:
                args = []
                kargs = {}
                if 'args' in pars:
                    args = pars['args']
                if 'kargs' in pars:
                    kargs = pars['kargs']

                if item == 'Rectangle':
                    r = plt.Rectangle(*args,**kargs)
                    s.ax.add_patch(r)

    if ((animate is None) or (animate == 'track')) and (fname is not None):
        plt.savefig(fname,dpi=dpi)
    elif (animate == 'cont') and (fname is not None):
        # Because we already used ani.save() to write the file
        pass
    else:
        plt.show()


# Configure probes based on where the telescope is pointed, the IRIS
# rotator position angle, and the coordinates of stars assigned to each
# probe. As a start, print out something resembling DS9 region
# definitions in sky coordinates.
# Notes:
#   - the stars are provided in the same order as the probes. At a rotator
#     PA of 0 deg, they are ordered counter-clockwise starting with the
#     top probe
#   - if autoselect set, instead assign probes to supplied stars automatically
#   - if the probe assignment is invalid, an exception will be thrown
#     by the Probe.set_cart() calls:
#     o ProbeLimits()       if it is not within the Probe's patrol range
#     o ProbeCollision()    if it would collide with another probe
#     o ProbeVignetteIFU()  if it would vignette the IFU pickoff
def oiwfs_sky(pointing,         # [ra,dec,PA] in degrees
              stars,            # [[ra,dec],[ra,dec],[ra,dec] in degrees
              autoselect=False  # automatically select stars if True
):

    # Initialize the OIWFS state
    s = State(None)

    # Set up WCS according to the pointing/instrument rotation
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [0, 0]  # center of the focal plane is tangent point
    w.wcs.crval = [pointing[0], pointing[1]]  # boresight coordinates
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    pa = np.radians(pointing[2]) # rotator angle
    w.wcs.cd = np.array([[-np.cos(pa),+np.sin(pa)], # includes RA sign flip
                         [np.sin(pa),np.cos(pa)]]) * \
                         (1./(platescale*3600.)) # deg/mm
        
    # Convert star coordinates to focal plane coordinates and store in State
    stars_fplane = w.wcs_world2pix(stars,1)
    for i in range(3):
        s.stars[i].x = stars_fplane[i][0]    # set star location
        s.stars[i].y = stars_fplane[i][1]
    
    if autoselect:
        # assign probes to stars automatically
        s.select_probes()
    else:
        # assign stars to probes in order
        for i in range(3):
            p = s.probes[i]
            p.star = s.stars[i]

    # Set probe positions to their assigned star
    for i in range(3):
        p = s.probes[i]
        p.set_cart(p.star.x,p.star.y)

    # print regions as lines and circles
    print "# OIWFS Regions specified in degrees in RA, Dec"
    for i in range(3):
        p = s.probes[i]                      # probe reference
        c_fplane = np.array([[p.x0,p.y0],[p.x,p.y]])
        c = w.wcs_pix2world(c_fplane,1)

        print "line (%f,%f,%f,%f)" % (c[0][0], c[0][1], c[1][0], c[1][1])
        print "circle (%f,%f,%f)" % (c[1][0],c[1][1],r_head/(platescale*3600.))


# -----------------------------------------------------------------------------

# Example code
        
if __name__ == '__main__':

    # Point OIWFS at a position on the sky.
    oiwfs_sky([180.,1,10],
              np.array([[180. + 0    ,1. + 0.006],
                        [180. + 0.012,1. - 0.006],
                        [180. - 0.012,1. - 0.006]]),
              autoselect=False)
    # sys.exit(1)

    # animate a sequence of random reconfigurations, show on-screen
    #run_sim(animate='cont',display=True,dwell=50,frameskip=1)

    # animate a sequence of pre-selected asterisms, show on-screen
    #run_sim(animate='cont',display=True,dwell=50,frameskip=1,
    #        aster=aster_seq,aster_select=True)

    
    # animated with vector field, write to file
    #run_sim(animate='cont',display=True,dwell=50,frameskip=1,
    #        plotlim=[-150,150,-150,150],
    #        i_ref=1,vectors=['att','rep','tran'],
    #        aster=aster_easy,aster_select=True,
    #        fname='reconfigs_vect.mp4',fps=60,frames=2895,dpi=150)
    #sys.exit(1)


    # animated non-sidereal tracking, write to file
    #run_sim(animate='cont',display=True,dwell=0,frameskip=1,
    #        plotlim=[-150,150,-150,150], star_vel=[0.5,2],
    #        aster=aster_move,aster_select=True)#,
    #        fname='nonsidereal.mp4',fps=60,frames=3500,dpi=150)
    #sys.exit(1)


    # animated non-sidereal tracking scrolling through catalog, show on-screen
    run_sim(animate='cont',display=True,dwell=0,frameskip=1,
            plotlim=[-150,150,-150,150], star_vel=[-2,0],
            catalog='stripe.txt',catalog_start=[0,0], aster_select=False)#,
            #fname='nonsidereal.mp4',fps=60,frames=3500,dpi=150)

    # --- Series of figures for SPIE paper --------------------------------
    #figtype = 'pdf'
    figtype = 'png'

    # overview plot
    run_sim(animate=None,aster=aster_easy,aster_select=True,
            aster_start=8,figsize=(5.5,5),dpi=150,fname='model.'+figtype)

    # show contours for normal situation
    plotlim=[-150,150,-150,150]
    
    contour_steps = 1
    levels = np.logspace(-1.8,0.2,40,endpoint=True)
    aster = [[ (-65,7), (-35,-15), (70,0) ]]
    startpos={'1':[-100,50]}
    run_sim(animate=None,aster=aster,aster_select=True,
            aster_start=0,fname='attract.'+figtype,plotlim=plotlim,
            i_ref=1, contours=['att'], levels=levels,
            startpos=startpos, contour_steps=contour_steps)

    run_sim(animate=None,aster=aster,aster_select=True,
            aster_start=0,fname='components.'+figtype,plotlim=plotlim,
            i_ref=1, contours=['att','rep'], levels=levels,
            startpos=startpos, contour_steps=contour_steps)

    run_sim(animate=None,aster=aster,aster_select=True,
            aster_start=0,fname='total.'+figtype,plotlim=plotlim,
            i_ref=1, contours=['tot'], levels=levels,
            startpos=startpos, contour_steps=contour_steps)

    # contours when there is a local minimum
    aster = [[ (0,50), (10,-65), (-35,-105) ]]
    startpos={'2':[-18,-20]}
    plotlim=[-90,40,-140,10]
    #contour_steps = 1
    levels = np.logspace(-3,0.0,60,endpoint=True)

    annotations = [ ('Rectangle',
                     {'args': [(-28,-41),30,12],
                      'kargs': {'angle': -20,
                                'color': 'm',
                                'fill' : False}})
                ]

    run_sim(animate=None,aster=aster,aster_select=True,
            aster_start=0,fname='localmin.'+figtype,
            i_ref=2, contours=['tot'], levels=levels,
            startpos=startpos, contour_steps=contour_steps,
            plotlim=plotlim,title_str='(a) Scalar Potential',
            annotations=annotations)

    # vector field plots
    #contour_steps=1
    run_sim(animate=None,aster=aster,aster_select=True,
            aster_start=0,fname='vect_basic.'+figtype,
            i_ref=2, vectors=['att','rep'], levels=levels,
            startpos=startpos, contour_steps=contour_steps,
            plotlim=plotlim, title_str='(b) Gradient Field',
            annotations=annotations)

    run_sim(animate=None,aster=aster,aster_select=True,
            aster_start=0,fname='vect_tran.'+figtype,
            i_ref=2, vectors=['tran'], levels=levels,
            startpos=startpos, contour_steps=contour_steps,
            plotlim=plotlim, title_str='(c) Transverse Component',
            annotations=annotations)

    run_sim(animate=None,aster=aster,aster_select=True,
            aster_start=0,fname='vect_tot.'+figtype,
            i_ref=2, vectors=['att','rep','tran'], levels=levels,
            startpos=startpos, contour_steps=contour_steps,
            plotlim=plotlim, title_str='(d) Total Vector Field',
            annotations=annotations)


    # Show some re-configurations
    plotlim=[-150,150,-150,150]
    for i in range(4):
        run_sim(animate='track',aster=aster_seq,aster_select=True,
                aster_start=i,fname='reconfig%i.%s' % (i,figtype),
                plotlim=plotlim)
