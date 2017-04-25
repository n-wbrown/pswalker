from __future__ import absolute_import
############
# Standard #
############
import inspect
import logging
from collections import namedtuple
###############
# Third Party #
###############
import numpy  as np
import pandas as pd
from scipy.optimize    import curve_fit
from bluesky.callbacks import LivePlot, LiveTable, CallbackBase
##########
# Module #
##########

logger = logging.getLogger(__name__)

Measurement = namedtuple('Measurement', ['alpha_0', 'alpha_1', 'centroid'])

class Model:
    """
    A generic model for the beam trajectory

    Parameters
    ----------
    func : callable
        Model function. Must take a tuple for each mirror pitch as the first
        parameter, then an arbitrary number of constants to be determined by
        observation

    args :
        Initial conditions for constants

    Attributes
    ----------
    params : tuple
        Constants determined by fit

    measurements : list
        List of measurements stored bt the model

    Example
    -------
    For instance, if we were to make the simplest approximation for the two
    mirror system, we would end up with an equation of the following form where
    `p1` and `p2` are the pitches of each mirror and `x0`, `x1`, and `x2` are
    constants

    .. math::

        y = x0 + x1*p1 + x2*p2

    This model can be loaded as such:

    .. code::

        simple = Model(lambda x,a,b,c : a + b*x.alpha_0 + c*x.alpha_1)
    """
    def __init__(self, func, *args):
        self.func         = func
        self.params       = args
        self.measurements = list()

        #Check that the initial conditions match function signature
        if self.params:
            sig = inspect.signature(self.func)
            if len(sig.parameters) != len(args)+1:
                raise ValueError('Invalid number of inital conditions')


    def add_measurement(self, measurement):
        """
        Add an additional measurement to the model
        
        Parameters
        ----------
        """
        self.measurements.append(measurement)


    def fit(self):
        """
        Use a table of sampled information to fit the model
        """
        #Can't fit without data
        if not self.measurements:
            raise ValueError("No measurements have been given to the model")
        #Unzip stored measurements
        results = [m.centroid for m in self.measurements]
        #Fit functions
        self.params, pcov = curve_fit(self.func, self.measurements,
                                      results, p0=self.params)
        return self.params, np.sqrt(np.diag(pcov))


    def estimate(self, position):
        """
        Produce an estimate based on the current model parameters

        If the model has not be :meth:`.fit`, this will be run before
        calculating the result

        Parameters
        ----------
        position : :class:`.Measurement`
            Measurement containing required information to evaluate
            :attr:`.func`
        """
        #Execute fit if none has been done
        if not self.params:
            self.fit()

        return self.func(positions, *self.params)


    def clear(self):
        """
        Clear prior fitted parameters from database
        """
        self.params = None
        self.measurements.clear()


class Monitor(CallbackBase):
    """
    Monitor class that writes the EPICS data to some place that can is
    retrievable by the other components.

    Whenever it receives the okay from walker, it is supposed to save the alpha
    positions of the mirrors and as well as the beam position at whichever imager
    is inserted.

    It should then be able to export the data as a pandas DataFrame or some way
    that makes querying for info easy.
    
    Attributes
    ----------
    use_cache:

    models:
    """
    def __init__(self, alpha_0, alpha_1):
        #Target field names for mirrors
        self.alpha_0, self.alpha_1 = alpha_0, alpha_1
        self.use_cache = True
        self.models    = dict()


    @property
    def all_models(self):
        """
        A list of all subscribed models
        """
        return [m for l in self.models.items() for m in l]
   

    def start(self, doc):
        """
        Start a new run
        """
        #Clear models
        if not self.use_cache and self.models:
            list(map(lambda x : x.clear(), self.all_models))
        super().start(doc)


    def event(self, doc):
        """
        Read an event emitted from the RunEngine
        """
        for centroid, models in self.models.items():
            try:
                measure = Measurement(doc['data'][self.alpha_0],
                                      doc['data'][self.alpha_1],
                                      doc['data'][centroid])

                for model in models:
                    model.add_measurement(measure)
                    model.fit()

            except KeyError:
                return

        self.update_plot()
        super().event(doc)
    
   
    def update_plot(self):
        """
        Update the plot
        """
        pass


    def fit(self):
        """
        Fit the stored information to the given model store
        """
        pass


    def subscribe(self, model, target_yag):
        """
        Subscribe a model to receive measurements

        Parameters
        ----------
        model : :class:`.Model`
            Model to receive measurement events

        target_yag : str
            Field to save as the centroid of beam
        """
        #If there are no other models attached
        if target_yag not in self.models:
            self.models[target_yag] = [model]

        else:
            self.models.append(model)

    def stop(self, doc):
        super().stop(doc)















    def __init__(self, **kwargs):
        self._all_data = pd.DataFrame()

        self._cached_centroids = np.zeros(2)
        self._cached_alphas = np.zeros(2)

    def get_all_data(self):
        """Return the full contents of the log."""
        pass
    
    def get_last_n_points(self, n):
        """Returns the last n entries of data."""
        pass

    def update(self):
        """
        Saves the current positions of the mirrors and the beam position on
        whatever imager is inserted.
        """
        self._new_data = True
        # Do the update
        pass

    @property
    def current_centroids(self):
        """
        Get the most recent entry of centroids and return as a numpy array.

        Use a simple caching system so as to not recompute the same
        """
        if self._new_centroids:
            # Grab most recent entries to the dataframe
            # centroids = self._all_data[['cent1','cent2']].tail(1)
            # self._cached_centroids = np.array(centroids)
            self._new_centroids = False
        return self._cached_centroids
       
    @property
    def current_alphas(self):
        """
        Get the most recent entry of mirror pitches and return as a numpy array. 
        """
        if self._new_alphas:
            # Grab most recent entries to the dataframe
            # alphas = self._all_data[['alpha1','alpha2']].tail(1)
            # self._cached_alphas = np.array(alphas)
            self._new_alphas = False
        return self._cached_alphas

