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

    def load_traj(self, sub_df):
        """
        load patient trajectory by id
        input :
        sub_df = dataframe of single patient
        """
        times    = sub_df['Weeks'].to_numpy()
        obs      = sub_df['Percent'].to_numpy()
        traj =  (times[:]-min(times[:])+0.1,obs[:])
        self. traj_p = traj
        return None

    def load_patient(self,df):
        """
        load patient by id
        input :
        df = dataframe of all patients
        """
        sub_df = df[df['Patient'] == self.id]

        self.load_traj(sub_df)
        self.age = set(sub_df['Age'])
        self.sex = set(sub_df['Sex'])
        self.smoker = set(sub_df['SmokingStatus'])
        self.base_fvc = max(sub_df['FVC'])*100/max(sub_df['Percent'])
        return None



