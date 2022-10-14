
class TimeEstimator(object):
    def __init__(self, job, profile_info):
        self.job = job
        self.profile_info = profile_info
        
    
    def optimistic_estimate(self, policy):
        assert isinstance(policy, list)
        for history in self.profile_info:
            history_policy, history_speed = history
            if history_policy == policy:
                return history_speed
        
        return min([history[1] for history in self.profile_info])
        

    def pessimistic_estimate(self, policy):
        for history in self.profile_info:
            history_policy, history_speed = history
            if history_policy == policy:
                return history_speed
        return min([history[1] for history in self.profile_info])
        raise NotImplementedError

    
