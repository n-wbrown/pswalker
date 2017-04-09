"""
Walker module that contains the class that allows the various walkers to 
interface with the hardware components.
"""

from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import utils.exceptions as uexc

from epics import caget
from beamDetector.detector import Detector
from utils.cvUtils import to_uint8, plot_image

################################################################################
#                                 Walker Class                                 #
################################################################################

class Walker(object):
    """
    Walker class that will actually perform the motions proposed by iterwalk or
    modelwalk. High level methods will call lightpath functions for lower level
    functionality (insert, remove and various checks).
    """

    def __init__(self, monitor, **kwargs):
        # This should be linked somehow to the templated models. A simple idea
        # is for walker to be a class that (dynamically?) inherits from the
        # model template that represents the system. For now I kept them separate
        # to avoid "premature optimization"
        self.monitor = monitor
        self.source = kwargs.get("source", Source())
        self.mirror_1 = kwargs.get("mirror_1", FlatMirror())
        self.mirror_2 = kwargs.get("mirror_2", FlatMirror())
        self.imager_1 = kwargs.get("imager_1", Imager())
        self.imager_2 = kwargs.get("imager_2", Imager())
        self.p1 = kwargs.get("p1", 0)   #Desired point at imager 1
        self.p2 = kwargs.get("p2", 0)   #Desired point at imager 2
        self.tolerance = kwargs.get("tolerance", 1e-6)

    def _move_alpha(self, new_alpha, mirror):
        """Performs the necessary steps to do a move of mirror 1."""
        alpha = self.monitor.current_alphas[mirror]
        # Only perform the move if it is to a position outside the tolerance
        if new_alpha < alpha-self.tolerance or new_alpha > alpha+self.tolerance:
            # There are two objectives for a move made by walker:
            # 1) Get to the goal
            # 2) Archive data from the move

            # The first is simple. A proposal for the second would be to do a
            # scan of n points between current_alpha1 and new_alpha1 and then at
            # each point store alpha1, alpha2 and pixel position on the
            # imager(s).

            # This seems like a slight variation on some of the canonical
            # bluesky scan demo so if that isn't what we are doing, why not?
            raise NotImplementedError

    def move_alphas(self, new_alpha_1, new_alpha_2):
        """Moves mirrors 1 and 2 to the inputted pitches."""
        self._move_alpha(new_alpha_1, 0)
        self._move_alpha(new_alpha_2, 1)

    def move_rel_alphas(self, rel_alpha_1, rel_alpha_2):
        """Moves mirrors 1 and 2 by the inputted pitches."""
        alphas = self.monitor.current_alphas
        self._move_alpha(alphas[0] + rel_alpha_1, 0)
        self._move_alpha(alphas[1] + rel_alpha_2, 1)

    def _jog_alpha_to_pixel(self, new_pixel, mirror):
        """
        Jogs the inputted mirror pitch until the beam centroid reaches the
        desired pixel.
        """
        current_centroid = self.monitor.current_centroids[mirror]
        # Only perform the move if it is to a different pixel than where it is
        if current_centroid != new_pixel:
            # Determine which direction to jog
            # Set up a monitor on the pv for beam centroidx
            # Run twf or twr depending on direction until pv monitor reads the
            # same centroid pixel as desired one
            raise NotImplementedError
        
    def jog_alphas_to_pixels(self, new_pixel_1, new_pixel_2):
        """
        Jogs each mirror until the centroid on the respective imager reaches
        the desired pixel.
        """
        self._jog_alpha_to_pixel(new_pixel_1, 0)
        self._jog_alpha_to_pixel(new_pixel_2, 1)

################################################################################
#                                 Imager Class                                 #
################################################################################

class Imager(object):
    """
    Imager object that will encapsulate the various yag screens along the
    beamline. 
    """
    def __init__(self, **kwargs):
        self.detector = kwargs.get("det", Detector(prep_mode="clip"))
        self.image    = None
        self.image_xsz = 0
        self.image_ysz = 0
        self.centroid = None
        self.bounding_box = None
        self.beam = False
        self.inserted = False
        self.mppix = kwargs.get("mppix", 1.25e-5) # meters per pixel
        self.pv_yag = kwargs.get("yag", None)
        self.pv_camera = kwargs.get("camera", None)
        self.simulation = kwargs.get("simulation", False)

    def get_image(self, norm="clip"):
        """Get a new image from the imager."""
        if self.simulation:
            try:
                uint_norm = to_uint8(self.image, "norm")
                self.image_ysz, self.image_xsz  = self.image.shape
                return to_uint8(uint_norm, "clip")
            except TypeError:
                self.sum = self.image.sum()
                return self.get_image(norm=norm)
        else:
            self.image = to_uint8(caget(self.pv_camera), norm)
            return self.image

    def get_beam(self, norm="clip", cent=True, bbox=True):
        """Return beam info (centroid and bounding box) of the saved image."""
        try:
            self.centroid, self.bounding_box = self.detector.find(self.image)
            self.beam = True
            if cent and bbox:
                return self.centroid, self.bounding_box
            elif cent:
                return self.centroid
            elif bbox:
                return self.bounding_box
            else:
                raise Exception
        except IndexError:
            self.beam = False
            return None
        
    def get_centroid(self, norm="clip"):
        """Return the centroid of the stored image."""
        return self.get_beam(norm, cent=True, bbox=False)

    def get_bounding_box(self, norm="clip"):
        """Return the bounding box of the stored image."""
        return self.get_beam(norm, cent=False, bbox=True)
        
    def get_image_and_centroid(self, norm="clip"):
        """Get a new image and return the centroid."""
        self.get_image(norm)
        return self.get_centroid(self, norm)

    def get_image_and_bounding_box(self, norm="clip"):
        """Get a new image and return the bounding box."""
        self.get_image(norm)
        return self.get_bounding_box(self, norm)

    def get_image_and_beam(self, norm="clip"):
        """Get a new image and return both the centroid and bounding box."""
        self.get_image(norm)
        return self.get_beam(self, norm)

    def insert(self):
        """Moves the yag to the inserted position."""
        # This will be filled in with lightpath functions/methods.
        raise NotImplementedError
    
    def remove(self):
        """Moves the yag to the removed position."""
        # This will be filled in with lightpath functions/methods.
        raise NotImplementedError

################################################################################
#                                 Mirror Class                                 #
################################################################################

class FlatMirror(object):
    """
    Mirror class to encapsulate the two HOMS (or any) mirrors.
    """

    def __init__(self, **kwargs):
        self._x          = kwargs.get("x", None)
        self._x_offset   = kwargs.get("x_offset", 0)
        self.alpha       = kwargs.get("alpha", None)
        self.z           = kwargs.get("z", None)
        self.x_pv        = kwargs.get("x_pv", None)
        self.alpha_pv    = kwargs.get("alpha_pv", None)
        self.simulation  = kwargs.get("simulation", True)
        # self.pos         = np.array([self.z, self.x])

        self._check_args()
        
        if self._x is None and not self.simulation:
            self.x = caget(self.x_pv) + self.x_offset

    def _check_args(self):
        # Only allowed to set alpha in sim mode
        if self.alpha is not None and not self.simulation:
            raise uexc.ImagerInputError(
                "Can only set alpha in simulation mode.")
        # Must set x motor pv if not in sim mode. Warning if set in sim mode.
        if self.x_pv is None and not self.simulation:
            raise uexc.ImagerInputError(
                "Must input x motor pv when not in simulation mode.")
        elif self.x_pv and self.simulation:
            warnings.warn("Ignoring input - X motor pv inputted when simulation \
mode is active.")
        # Must set alpha motor pv not in sim mode. Warning if set in sim mode.
        if self.alpha_pv is None and not self.simulation:
            raise uexc.ImagerInputError(
                "Must input alpha motor pv when not in simulation mode.")
        elif self.alpha_pv and self.simuation:
            warnings.warn("Ignoring input - alpha motor pv inputted when \
simulation mode is active.")

    @property
    def x(self):
        if self.simulation:
            return self._x + self._x_offset
        else:
            return caget(self.x_pv) + self._x_offset
    @x.setter
    def x(self, val):
        if self.simulation:
            self._x = val - self._x_offset
        else:
            caput(self.x_pv, val - self._x_offset)

    # @property
    # def x_offset(self):
    #     return self._x_offset
    
    # @x_offset.setter
    # def x_offset(self, val):
    #     self._x_offset = val
    #     if not
        

################################################################################
#                                 Source Class                                 #
################################################################################

class Source(object):
    def __init__(self):
        # I believe there is a Linac class somewhere in blinst but I don't know
        # how well it works or it is something we even want to use. It could be
        # too low level for what we trying to do.
        pass
