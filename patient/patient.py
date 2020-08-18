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
    def __init__(self,id,age,smoker,sex,traj_p):
        super().__init__()

        self.id     =  id
        self.age    = age
        self.smoker = smoker
        self.sex    = sex
        