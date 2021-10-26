# -*- coding: utf-8 -*-

from ModelSelectionStandard import ModelSelectionStandard
import pandas as pd
import numpy as np


class ModelSelectionDeltaAIC(ModelSelectionStandard):
    """
    Model Selection problem with tabu algorithm (Short memory version) \n
    Features: Deﬁne a new attribute that signals when the AIC change exceeds some value. \n
    
    Args: \n
        tenure: The number of iterations over which an attribute is tabu \n
        max_iteration: The max number of iterations \n
        file_path: The file path of the dataset \n
        threshold_aic: Let the attribute equal 1 if delta AIC exceeds this value. Default: 0.5 \n
        seed: Random number seed. Default: 1234
    """
    
    def __init__(self, tenure, max_iteration, file_path, threshold_aic = 0.5, seed = 1234):
        super().__init__(tenure, max_iteration, file_path, seed)
        self.threshold_aic = threshold_aic
        
    def attribute_aic(self, delta_aic):
        """
        Return 1 if delta AIC exceeds threshold value, else return 0
        """
        if delta_aic > self.threshold_aic:
            return 1
        else:
            return 0
        
    def tabu_search(self):
        """
        Implementation Tabu search algorithm \n
        Tabu attribute: AIC change exceeds given threshold \n
        Aspiration criteria: Permit a tabu move if it provides a higher value 
        of the objective function than has been found in any iteration so far \n
        
        Returns: \n
            history_solutions: list \n
            history_aic: list
        """
        # Initiate
        tabu_structure = self.get_tabu_structure()
        best_solution = self.get_initial_solution()
        best_obj_value = self.obj_fun(best_solution)
        current_solution = best_solution
        current_obj_value = best_obj_value
        iteration = 1
        neighborhood_not_empty = True
        history_solutions = []
        history_solutions.append(current_solution)
        history_aic = []
        history_aic.append(current_obj_value)
        
        # Main logic
        while iteration <= self.max_iteration and neighborhood_not_empty:
            # Get neighborhood and calculate corresponding AIC
            current_neighborhood = self.get_neighborhood(current_solution)
            current_neighborhood_lst = []
            for one_neighborhood in current_neighborhood:
                candidate_solution = current_solution.copy()
                candidate_solution[one_neighborhood[0][0]] = one_neighborhood[0][1]
                current_neighborhood_lst.append([one_neighborhood[0], candidate_solution, self.obj_fun(candidate_solution)])
            current_neighborhood_df = pd.DataFrame(current_neighborhood_lst, columns = ["Neighborhood", "Candidate", "AIC"])
            current_neighborhood_df = current_neighborhood_df.sort_values(by = ["AIC"])
            
            # Check candidates
            for i_row in range(current_neighborhood_df.shape[0]):
                best_candidate_obj_value = current_neighborhood_df.iloc[i_row, 2]
                delta_aic = best_candidate_obj_value - history_aic[-1]
                best_candidate = (current_neighborhood_df.iloc[i_row, 0][0], self.attribute_aic(delta_aic))
                not_tabu = True
                
                # In tabu list
                if tabu_structure[best_candidate] >= iteration:
                    not_tabu = False
                    # Check the aspiration criterion
                    if best_candidate_obj_value <= best_obj_value:
                        if i_row == (current_neighborhood_df.shape[0] - 1):
                            neighborhood_not_empty = False
                    else:
                        not_tabu = True
                
                # Make the move
                if not_tabu:
                    current_solution = current_neighborhood_df.iloc[i_row, 1]
                    current_obj_value = best_candidate_obj_value
                    history_solutions.append(current_solution)
                    history_aic.append(current_obj_value)
                    # Update best solution
                    if current_obj_value > best_obj_value:
                        best_solution = current_solution
                        best_obj_value = current_obj_value
                    # Update tabu list
                    update = (best_candidate[0], np.abs(best_candidate[1] -1))
                    tabu_structure[update] = iteration + self.tenure
                    break
            
            iteration += 1
            
        # Output
        return history_solutions, history_aic