# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class Template(ABC):
    """
    A template for tabu algorithm
    """
    
    @abstractmethod
    def input_data(self):
        """
        Input dataset
        """
        pass
    
    @abstractmethod
    def get_tabu_structure(self):
        """
        Return a dict of tabu
        """
        pass
    
    @abstractmethod
    def get_initial_solution(self):
        """
        Return the intial solution
        """
        pass
    
    @abstractmethod
    def get_neighborhood(self, current_solution):
        """
        Return the neighborhood of current solution
        """
        pass
        
    @abstractmethod
    def obj_fun(self, solution):
        """
        Define the objective function
        """
        pass
        
    @abstractmethod
    def tabu_search(self):
        """
        Implementation Tabu search algorithm
        """
        pass