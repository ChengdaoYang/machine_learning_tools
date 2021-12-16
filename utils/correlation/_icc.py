"""
implementation of Intraclass correlations
"""
import numpy as np


class IntraClassCorrelationCoefficient:
    def __init__(self, groups, group_axis=0):
        """
        Computer variance icc, but only icc1 and fish icc are useful for 
        correlation purpose,
        Please see paper:
        ....

        Parameters
        ----------
        groups : List of iterable, 
            List of data of each group. 2d data with each row or col represent
            a group, and another axis represent index of item in group.
            Note: groups do not need to be same size, and match index in pairs
            as ICC1 computation outlined.

        group_axis: int.  Default=0.
            axis with represent the gorup.(to be developed in future for more
            flexible input)
            
        """
        self.group_axis = group_axis
        self.groups = groups


    def __str__(self):
        pass

    def __doc__(self):
        pass


    @property
    def group_axis(self):
        return self._group_axis


    @group_axis.setter
    def group_axis(self, group_axis):
        if isinstance(group_axis, int):
            raise ValueError("group_axis take Integer only")
        self._group_axis = group_axis


    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, groups):
        if not isinstance(groups, list):
            raise TypeError(f"groups must be list, not {type(group)}")
        if len(groups) <= 1:
            raise ValueError("Must contain more than 2 groups for icc calulation")
        groups = [np.asarray(g) for g in groups]
        if any(g.shape[self.group_axis] == 0 for g in groups):
            raise ValueError("At least one group is empty, group must contain ")
        self._groups = groups





    @property
    def total_dof(self):
        raise NotImplementedError("not yet implemented")

    @property
    def between_group_dof(self):
        raise NotImplementedError("not yet implemented")

    @property
    def within_group_dof(self):
        raise NotImplementedError("not yet implemented")


    def group_means(self):
        """ Return a list of means for each group"""
        return [np.mean(arr) for arr in self.groups]


    def group_mean_centered_groups(self):
        """ Return a list of ndarray of group mean centered data for each group"""

        return [np.subtract(arr,arr.mean()) for arr in self.groups]




    def icc_fisher(self):
        raise NotImplementedError("not yet implemented")


    def icc1(self):
        """
        Not correctly implemented bias icc1, based on sigma_b/(sigma_b + sigma_w)

        ICC reveal percentage of variance bw group in total variance.

        Note: this is not the same ICC1 as in Shrout & Fleiss' paper.
        """
        # Computer means of each group in the list, and combine them into array
        between_group_var = np.array(
            self.group_means()
        ).var(ddof=0)  # ddof may be wrong

        # Cluster mean centered data to remove between group effect,
        # then computer its var with left with only within group effect
        # concatenate can aid in handle difference length of group data
        within_group_var = np.concatenate(
            self.group_mean_centered_groups()
        ).var(ddof=0)  # ddof may be wrong

        return between_group_var/(within_group_var + between_group_var)


        
        

