# file with dictionaries containing the required information for the columns

# Dictionary with the future names as keys and the column numbers for those features in the modified data sets as values
jet_feautures = {
    0: {'DER_mass_transverse_met_lep': 0, 'DER_mass_vis': 1, 'DER_pt_h': 2, 'DER_deltar_tau_lep': 3,
        'DER_pt_tot': 4, 'DER_sum_pt': 5, 'DER_pt_ratio_lep_tau': 6, 'DER_met_phi_centrality': 7,
        'PRI_tau_pt': 8, 'PRI_tau_eta': 9, 'PRI_tau_phi': 10, 'PRI_lep_pt': 11, 'PRI_lep_eta': 12, 
        'PRI_lep_phi': 13, 'PRI_met': 14, 'PRI_met_phi': 15, 'PRI_met_sumet': 16},
    1: {'DER_mass_MMC': 0,'DER_mass_transverse_met_lep': 1, 'DER_mass_vis': 2, 'DER_pt_h': 3, 'DER_deltar_tau_lep': 4,
        'DER_pt_tot': 5, 'DER_sum_pt': 6, 'DER_pt_ratio_lep_tau': 7, 'DER_met_phi_centrality': 8,
        'PRI_tau_pt': 9, 'PRI_tau_eta': 10, 'PRI_tau_phi': 11, 'PRI_lep_pt': 12, 'PRI_lep_eta': 13, 
        'PRI_lep_phi': 14, 'PRI_met': 15, 'PRI_met_phi': 16, 'PRI_met_sumet': 17},
    2: {'DER_mass_transverse_met_lep': 0,'DER_mass_vis': 1,'DER_pt_h': 2,'DER_deltar_tau_lep': 3,
        'DER_pt_tot': 4,'DER_sum_pt': 5,'DER_pt_ratio_lep_tau': 6,'DER_met_phi_centrality': 7,'PRI_tau_pt': 8,
        'PRI_tau_eta': 9,'PRI_tau_phi': 10,'PRI_lep_pt': 11,'PRI_lep_eta': 12,'PRI_lep_phi': 13,'PRI_met': 14,
        'PRI_met_phi': 15,'PRI_met_sumet': 16,'PRI_jet_leading_pt': 17,'PRI_jet_leading_eta': 18,'PRI_jet_leading_phi': 19,
        'PRI_jet_all_pt': 20},
    3: {'DER_mass_MMC': 0,'DER_mass_transverse_met_lep': 1,'DER_mass_vis': 2,'DER_pt_h': 3,'DER_deltar_tau_lep': 4,
        'DER_pt_tot': 5,'DER_sum_pt': 6,'DER_pt_ratio_lep_tau': 7,'DER_met_phi_centrality': 8,'PRI_tau_pt': 9,
        'PRI_tau_eta': 10,'PRI_tau_phi': 11,'PRI_lep_pt': 12,'PRI_lep_eta': 13,'PRI_lep_phi': 14,'PRI_met': 15,
        'PRI_met_phi': 16,'PRI_met_sumet': 17,'PRI_jet_leading_pt': 18,'PRI_jet_leading_eta': 19,'PRI_jet_leading_phi': 20,
        'PRI_jet_all_pt': 21},
    4: {'DER_mass_transverse_met_lep': 0, 'DER_mass_vis': 1, 'DER_pt_h': 2, 'DER_deltaeta_jet_jet': 3, 
        'DER_mass_jet_jet': 4, 'DER_prodeta_jet_jet': 5, 'DER_deltar_tau_lep': 6, 'DER_pt_tot': 7,
        'DER_sum_pt': 8, 'DER_pt_ratio_lep_tau': 9, 'DER_met_phi_centrality': 10, 'DER_lep_eta_centrality': 11,
        'PRI_tau_pt': 12, 'PRI_tau_eta': 13, 'PRI_tau_phi': 14, 'PRI_lep_pt': 15, 'PRI_lep_eta': 16,
        'PRI_lep_phi': 17, 'PRI_met': 18, 'PRI_met_phi': 19, 'PRI_met_sumet': 20, 'PRI_jet_leading_pt': 21,
        'PRI_jet_leading_eta': 22, 'PRI_jet_leading_phi': 23, 'PRI_jet_subleading_pt': 24, 
        'PRI_jet_subleading_eta': 25, 'PRI_jet_subleading_phi': 26, 'PRI_jet_all_pt': 27},
    5: {'DER_mass_MMC': 0,'DER_mass_transverse_met_lep': 1, 'DER_mass_vis': 2, 'DER_pt_h': 3, 'DER_deltaeta_jet_jet': 4, 
        'DER_mass_jet_jet': 5, 'DER_prodeta_jet_jet': 6, 'DER_deltar_tau_lep': 7, 'DER_pt_tot': 8,
        'DER_sum_pt': 9, 'DER_pt_ratio_lep_tau': 10, 'DER_met_phi_centrality': 11, 'DER_lep_eta_centrality': 12,
        'PRI_tau_pt': 13, 'PRI_tau_eta': 14, 'PRI_tau_phi': 15, 'PRI_lep_pt': 16, 'PRI_lep_eta': 17,
        'PRI_lep_phi': 18, 'PRI_met': 19, 'PRI_met_phi': 20, 'PRI_met_sumet':21, 'PRI_jet_leading_pt': 23,
        'PRI_jet_leading_eta': 24, 'PRI_jet_leading_phi': 25, 'PRI_jet_subleading_pt': 26, 
        'PRI_jet_subleading_eta': 27, 'PRI_jet_subleading_phi': 28, 'PRI_jet_all_pt': 29},
    6: {'DER_mass_transverse_met_lep': 0, 'DER_mass_vis': 1, 'DER_pt_h': 2, 'DER_deltaeta_jet_jet': 3, 
        'DER_mass_jet_jet': 4, 'DER_prodeta_jet_jet': 5, 'DER_deltar_tau_lep': 6, 'DER_pt_tot': 7,
        'DER_sum_pt': 8, 'DER_pt_ratio_lep_tau': 9, 'DER_met_phi_centrality': 10, 'DER_lep_eta_centrality': 11,
        'PRI_tau_pt': 12, 'PRI_tau_eta': 13, 'PRI_tau_phi': 14, 'PRI_lep_pt': 15, 'PRI_lep_eta': 16,
        'PRI_lep_phi': 17, 'PRI_met': 18, 'PRI_met_phi': 19, 'PRI_met_sumet': 20, 'PRI_jet_leading_pt': 21,
        'PRI_jet_leading_eta': 22, 'PRI_jet_leading_phi': 23, 'PRI_jet_subleading_pt': 24, 
        'PRI_jet_subleading_eta': 25, 'PRI_jet_subleading_phi': 26, 'PRI_jet_all_pt': 27},
    7: {'DER_mass_MMC': 0,'DER_mass_transverse_met_lep': 1, 'DER_mass_vis': 2, 'DER_pt_h': 3, 'DER_deltaeta_jet_jet': 4, 
        'DER_mass_jet_jet': 5, 'DER_prodeta_jet_jet': 6, 'DER_deltar_tau_lep': 7, 'DER_pt_tot': 8,
        'DER_sum_pt': 9, 'DER_pt_ratio_lep_tau': 10, 'DER_met_phi_centrality': 11, 'DER_lep_eta_centrality': 12,
        'PRI_tau_pt': 13, 'PRI_tau_eta': 14, 'PRI_tau_phi': 15, 'PRI_lep_pt': 16, 'PRI_lep_eta': 17,
        'PRI_lep_phi': 18, 'PRI_met': 19, 'PRI_met_phi': 20, 'PRI_met_sumet':21, 'PRI_jet_leading_pt': 23,
        'PRI_jet_leading_eta': 24, 'PRI_jet_leading_phi': 25, 'PRI_jet_subleading_pt': 26, 
        'PRI_jet_subleading_eta': 27, 'PRI_jet_subleading_phi': 28, 'PRI_jet_all_pt': 29}
}

# Dictionary with the feature names as keys and the column numbers for those features in the unmodified data set as values
feautures = {"DER_mass_MMC":0,"DER_mass_transverse_met_lep":1,"DER_mass_vis":2,"DER_pt_h":3,"DER_deltaeta_jet_jet":4,"DER_mass_jet_jet":5,"DER_prodeta_jet_jet":6,
	     "DER_deltar_tau_lep":7,"DER_pt_tot":8,"DER_sum_pt":9,"DER_pt_ratio_lep_tau":10,"DER_met_phi_centrality":11,"DER_lep_eta_centrality":12,"PRI_tau_pt":13,
             "PRI_tau_eta":14,"PRI_tau_phi":15,"PRI_lep_pt":16,"PRI_lep_eta":17,"PRI_lep_phi":18,"PRI_met":19,"PRI_met_phi":20,"PRI_met_sumet":21,"PRI_jet_num":22,
             "PRI_jet_leading_pt":23,"PRI_jet_leading_eta":24,"PRI_jet_leading_phi":25,"PRI_jet_subleading_pt":26,"PRI_jet_subleading_eta":27,"PRI_jet_subleading_phi":28,
             "PRI_jet_all_pt":29}
