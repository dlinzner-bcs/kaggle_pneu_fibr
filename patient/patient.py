import pandas as pd

class patient():
    """
    patient class.


    Parameters
    ----------
    id      = identification number
    age     = age
    smoker  = smoker never smoked/ ex smoker / current smoker
    sex     = sex female/male
    traj_p  = percentage observation trajectories
    """
    def __init__(self,id):
        super().__init__()

        self.id     =  id

    def load_traj(self, subframe):
        times    = subframe['Weeks'].to_numpy()
        obs      = subframe['Percent'].to_numpy()
        traj =  (times[:]-min(times[:])+0.1,obs[:])
        self. traj_p = traj
        return None

    def load_patient(self,dataframe):
        subframe = dataframe[dataframe['Patient'] == self.id]

        self.load_traj(subframe)
        self.age = set(subframe['Age'])
        self.sex = set(subframe['Sex'])
        self.smoker = set(subframe['SmokingStatus'])
        return None


