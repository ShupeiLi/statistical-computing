# -*- coding: utf-8 -*-

from ModelSelectionStandard import ModelSelectionStandard
import pandas as pd
import statsmodels.api as sm
import numpy as np


class ModelSelectionInfluence(ModelSelectionStandard):
    """
    Model Selection problem with tabu algorithm (Short memory version) \n
    Features: Aspiration by influence. Measure influence with changes in adjusted R-square \n
    
    Args: \n
        tenure: The number of iterations over which an attribute is tabu \n
        max_iteration: The max number of iterations \n
        file_path: The file path of the dataset \n
        threshold_r2: Regard a move as a high-influence move if delta R2 (adjusted) exceeds this value. Default: 0.01 \n
        seed: Random number seed. Default: 1234
    """
    
    def __init__(self, tenure, max_iteration, file_path, threshold_r2 = 0.01, seed = 1234):
        super().__init__(tenure, max_iteration, file_path, seed)
        self.threshold_r2 = threshold_r2
        
    def r2(self, solution):
        """
        Calculate adjusted R2
        """
        x_index = [item[0] for item in filter(lambda x: x[1] == 1, list(enumerate(solution)))]
        x_selected = sm.add_constant(self.x_full[:, x_index])
        model = sm.OLS(self.y, x_selected).fit()
        return model.rsquared_adj
        
    def tabu_search(self):
        """
        Implementation Tabu search algorithm \n
        Tabu attribute: Add/delete one predictor \n
        Aspiration criteria: Aspiration by influence. Measure influence with changes in adjusted R-square \n
        
        Returns: \n
            history_solutions: list \n
            history_solutions_AIC: list
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
                best_candidate = current_neighborhood_df.iloc[i_row, 0]
                best_candidate_solution = current_neighborhood_df.iloc[i_row, 1]
                best_candidate_obj_value = current_neighborhood_df.iloc[i_row, 2]
                not_tabu = True
                
                # In tabu list
                if tabu_structure[best_candidate] >= iteration:
                    not_tabu = False
                    # Check the aspiration criterion
                    if len(history_solutions) >= 2:
                        previous_r2 = self.r2(history_solutions[-2])
                        current_r2 = self.r2(current_solution)
                        best_candidate_r2 = self.r2(best_candidate_solution)
                        if abs(current_r2 - previous_r2) > self.threshold_r2 and abs(best_candidate_r2 - current_r2) < self.threshold_r2:
                            not_tabu = True
                        elif i_row == (current_neighborhood_df.shape[0] - 1):
                            neighborhood_not_empty = False
                        else:
                            continue
                
                # Make the move
                if not_tabu:
                    current_solution = best_candidate_solution
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