# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:26:26 2023

@author: cpannetier

IMPLEMENTATIONS ALTERNATIVES DE LA FONCTION RELOCK

"""




""" Implementation de la fonction SEARCH telle que décrite pas Lacour """
""" La fonction sawtooth est commune a tous les télescopes """

# IgdRank = np.linalg.matrix_rank(simu.Igd[it])
# NotCophased = (IgdRank < NA-1)
# simu.IgdRank[it] = IgdRank

# if NotCophased:
#     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
    
#     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
#     # This situation could pose a problem but we don't manage it yet        
#     if (simu.time_since_loss[it] > config.FT['SMdelay']):
        
#         Igdna = np.dot(config.FS['OPD2Piston'],
#                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
        
#         # Fringe loss
#         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
        
#         # Photometry loss
#         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
            
#         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
#         simu.LossDueToInjection[it] = comparison.all()       # Evaluates if the two arrays are the same
        
#         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
#             config.FT['state'][it] = 1
            
#             if (config.FT['state'][it-1] == 0):         # Last frame, all telescopes were tracked
#                 config.FT['it0'] = it ; config.FT['it_last'] = it
#                 config.FT['LastPosition'] = 0#np.copy(config.FT['usaw'][it-1])
    
#             config.FT['usaw'][it] = searchfunction_basical(config.FT['usaw'][it-1], it)
        
#             Kernel = np.identity(NA) - Igdna
#             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
#             Kernel = np.dot(simu.NoPhotometryFiltration[it],Kernel)
        
#             config.FT['usearch'][it] = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
            
#             # Patch pour commander en incrément comme l'algo réel
#             SearchIncrement = config.FT['usearch'][it] - config.FT['usearch'][it-1]
            
#         else:
#             config.FT['state'][it] = 0
#             SearchIncrement = 0
#             print(it, "Loss due to injection")
#     else:
#         config.FT['state'][it] = 0
#         SearchIncrement = 0
#         print(it, "Delay short")
        
# else:
#     simu.time_since_loss[it] = 0
#     config.FT['state'][it] = 0
#     config.FT['eps'] = 1
    
#     SearchIncrement = 0
#     print(it, "Cophased")

# # if config.TELref:
# #     iTel = config.TELref-1
# #     SearchIncrement = SearchIncrement - SearchIncrement[iTel]

# SearchIncrement = config.FT['search']*SearchIncrement

# # The command is sent at the next time, that's why we note it+1
# usearch = simu.SearchCommand[it] + SearchIncrement

# simu.SearchCommand[it+1] = usearch

""" Implementation avec la fonction sawtooth spécifique à chaque télescope """

# IgdRank = np.linalg.matrix_rank(simu.Igd[it])
# NotCophased = (IgdRank < NA-1)
# simu.IgdRank[it] = IgdRank

# if NotCophased:
#     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
    
#     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
#     # This situation could pose a problem but we don't manage it yet        
#     if (simu.time_since_loss[it] > config.FT['SMdelay']):
        
#         Igdna = np.dot(config.FS['OPD2Piston'],
#                        np.dot(simu.Igd[it],config.FS['Piston2OPD']))
        
#         # Fringe loss
#         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
#         # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
        
#         # Photometry loss
#         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
            
#         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
#         simu.LossDueToInjection[it] = comparison.all()       # Evaluates if the two arrays are the same
        
#         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
#             config.FT['state'][it] = 1
            
#             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
#             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
#             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
            
#             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)
            
#             # if (config.FT['state'][it-1] == 0):         # Last frame, all telescopes were tracked
#             #     config.FT['it0'] = np.ones(NA)*it ; config.FT['it_last'] = np.ones(NA)*it
#             #     config.FT['LastPosition'] = np.copy(config.FT['usaw'][it-1])
                
#             # elif sum(TelescopesThatNeedARestart) > 0:
                
#             #     # Version "Restart only concerned telescopes" (06-10-2021)
#             #     # --> doesn't work because it avoids some OPDs.
#             #     # for ia in TelescopesThatNeedARestart:
#             #     #     config.FT['it0'][ia] = it ; config.FT['it_last'][ia] = it
#             #     #     config.FT['LastPosition'][ia] = 0
            
#             #     # Version "Restart all" (06-10-2021)
#             #     # Restart all telescope from their current position.
#             #     config.FT['it0'] = np.ones(NA)*it
#             #     config.FT['it_last'] = np.ones(NA)*it
#             #     config.FT['LastPosition'] = np.copy(config.FT['usaw'][it-1])
                
#             # config.FT['usaw'][it] = searchfunction(config.FT['usaw'][it-1])         # Fonction search de vitesse 1µm/frame par piston
                
#             # Kernel = np.identity(NA) - Igdna
#             # simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
#             # Kernel = np.dot(simu.NoPhotometryFiltration[it],Kernel)                 
            
#             # # After multiplication by Kernel, the OPD velocities can only be lower or equal than before
            
#             # usearch = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
        
#             if (config.FT['state'][it-1] == 0):# or (sum(TelescopesThatNeedARestart) >0) :
#                 config.FT['it0'] = it ; config.FT['it_last'] = it
#                 config.FT['LastPosition'] = config.FT['usaw'][it-1]
        
#             usaw = np.copy(config.FT['usaw'][it-1])
#             config.FT['usaw'][it] = searchfunction2(usaw,it)      # In this version, usaw is a float
        
#             Kernel = np.identity(NA) - Igdna
#             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
#             Kernel = np.dot(simu.NoPhotometryFiltration[it],Kernel)
        
#             usearch = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
            
#         else:
#             config.FT['state'][it] = 0
#             usearch = simu.SearchCommand[it]
#     else:
#         config.FT['state'][it] = 0
#         usearch = simu.SearchCommand[it]
        
# else:
#     simu.time_since_loss[it] = 0
#     config.FT['state'][it] = 0
#     # Version usaw vector
#     # config.FT['eps'] = np.ones(NA)
    
#     # Version usaw float
#     config.FT['eps'] = 1
    
#     usearch = simu.SearchCommand[it]
    
    
# # if config.TELref:
# #     iTel = config.TELref-1
# #     usearch = usearch - usearch[iTel]

# usearch = config.FT['search']*usearch
# # The command is sent at the next time, that's why we note it+1
# simu.SearchCommand[it+1] = usearch



""" New implementation RELOCK command 08/02/2022 """

# IgdRank = np.linalg.matrix_rank(simu.Igd[it])
# NotCophased = (IgdRank < NA-1)
# simu.IgdRank[it] = IgdRank

# if NotCophased:
#     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
    
#     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
#     # This situation could pose a problem but we don't manage it yet        
#     if (simu.time_since_loss[it] > config.FT['SMdelay']):
        
#         Igdna = np.dot(config.FS['OPD2Piston'],
#                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
        
#         CophasedBaselines=np.where(np.diag(simu.Igd[it])>0.5)[0]
#         CophasedPairs=[]
#         for ib in CophasedBaselines:
#             ia,iap = config.FS['ich'][ib][0], config.FS['ich'][ib][1]
#             CophasedPairs.append([ia,iap])
            
#         CophasedGroups = JoinOnCommonElements(CophasedPairs)
        
#         # Fringe loss
#         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
#         # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
        
#         # Photometry loss
#         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
            
#         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
#         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
        
#         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
#             config.FT['state'][it] = 1

#             # If it=0, initialize LastPosition to 0. 
#             # Else, it will remain the last value of SearchCommand, which has
#             # not change since last RELOCK state.
            
#             LastPosition = config.FT['LastPosition'][it]
            
#             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
#             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
#             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
            
#             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

#             if sum(TelescopesThatNeedARestart)>0:
#                 config.FT['it_last']=it ; #Ldico[ia]['eps']=1 #; Ldico[ia]['it0']=it ;   

#             usaw, change = searchfunction_inc_basical(it)
#             config.FT['usaw'][it]= usaw

#             Kernel = np.identity(NA) - Igdna
#             Increment = np.dot(Kernel,config.FT['usaw'][it]*config.FT['Velocities'])
#             Increment = Increment/np.ptp(Increment) * config.FT['maxVelocity']
            
#             if change:  # Change direction of scan
#                 # Fais en sorte que les sauts de pistons de télescopes cophasés 
#                 # entre eux maintiennent l'OPD constante: si 1 et 2 sont cophasés
#                 # avec OPD=p2-p1, au prochain saut le télescope 2 va à la position
#                 # du T1 + OPD et pas à la position qu'il avait avant le précédent saut.
#                 for group in CophasedGroups:    
#                     for ig in range(1,len(group)):
#                         ia = int(float(group[ig])-1) ; i0 = int(float(group[0])-1)
#                         LastPosition[ia] = LastPosition[i0] + simu.SearchCommand[it,ia]-simu.SearchCommand[it,i0]
#                 usearch = LastPosition + Increment
#                 LastPosition = simu.SearchCommand[it]
                
#             else:
#                 usearch = simu.SearchCommand[it]+Increment
                
#             config.FT['LastPosition'][it+1] = LastPosition
            
#             # You should send command only on telescope with flux
#             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
#             usearch = np.dot(simu.NoPhotometryFiltration[it],usearch)
        
        
#         else:
#             usearch = simu.SearchCommand[it]
    
#     else:
#         usearch = simu.SearchCommand[it]
        
# else:
#     simu.time_since_loss[it] = 0
#     usearch = simu.SearchCommand[it]
    
    
# usearch = config.FT['search']*usearch
# # The command is sent at the next time, that's why we note it+1
# simu.SearchCommand[it+1] = usearch

""" Implementation comme Sylvain sans réinitialisation """


# IgdRank = np.linalg.matrix_rank(simu.Igd[it])
# NotCophased = (IgdRank < NA-1)
# simu.IgdRank[it] = IgdRank


# if NotCophased:
#     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
    
#     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
#     # This situation could pose a problem but we don't manage it yet        
#     if (simu.time_since_loss[it] > config.FT['SMdelay']):
        
#         Igdna = np.dot(config.FS['OPD2Piston'],
#                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
        
#         Kernel = np.identity(NA) - Igdna
        
#         # Fringe loss
#         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
        
#         # Photometry loss
#         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
            
#         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
#         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
        
#         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
#             config.FT['state'][it] = 1
            
#             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
#             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
            
#             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

#             if config.FT['state'][it-1] != 1:
#                 config.FT['eps'] = np.ones(NA)
#                 config.FT['it0'] = np.ones(NA)*it
#                 config.FT['it_last'] = np.ones(NA)*it
            
#             Velocities = np.dot(Kernel,config.FT['Velocities'])
#             Increment = searchfunction_inc_sylvain(it, Velocities)    
        
#             #You should send command only on telescope with flux
#             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
#             Increment = np.dot(simu.NoPhotometryFiltration[it],Increment)
            
#             usearch = simu.SearchCommand[it] + Increment
            
#         else:
#                 Increment = np.zeros(NA)
        
#     else:
#         Increment = np.zeros(NA)
        
# else:
#     simu.time_since_loss[it] = 0
#     Increment = np.zeros(NA)
        
# Increment = config.FT['search']*Increment

# usearch = simu.SearchCommand[it] + Increment
# # The command is sent at the next time, that's why we note it+1
# simu.SearchCommand[it+1] = usearch



""" Implementation comme Sylvain avec réinitialisation """
""" PROBLEME: beaucoup d'OPD sont sautées """


# IgdRank = np.linalg.matrix_rank(simu.Igd[it])
# NotCophased = (IgdRank < NA-1)
# simu.IgdRank[it] = IgdRank


# if NotCophased:
#     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
    
#     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
#     # This situation could pose a problem but we don't manage it yet        
#     if (simu.time_since_loss[it] > config.FT['SMdelay']):
        
#         Igdna = np.dot(config.FS['OPD2Piston'],
#                         np.dot(simu.Igd[it],config.FS['Piston2OPD']))
        
#         Kernel = np.identity(NA) - Igdna
        
#         # Fringe loss
#         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
        
#         # Photometry loss
#         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
            
#         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
#         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
        
#         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
#             config.FT['state'][it] = 1
            
#             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
#             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
#             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
            
#             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)

#             if config.FT['state'][it-1] != 1:
#                 config.FT['eps'] = np.ones(NA)
#                 config.FT['it0'] = np.ones(NA)*it
#                 config.FT['it_last'] = np.ones(NA)*it


#             # print(TelescopesThatNeedARestart)
#             # print(config.FT['it_last'])
#             # print(config.FT['it0'])
#             for ia in range(NA):
#                 if ia in TelescopesThatNeedARestart:
#                     config.FT['eps'][ia] = 1
#                     config.FT['it_last'][ia] = it
#                     config.FT['it0'][ia] = it
#                     config.FT['LastPosition'][ia] = 0
            
#             Velocities = np.dot(Kernel,config.FT['Velocities'])
#             Increment = searchfunction_inc_sylvain(it, Velocities)    
        
#             #You should send command only on telescope with flux
#             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
#             Increment = np.dot(simu.NoPhotometryFiltration[it],Increment)
            
#             usearch = simu.SearchCommand[it] + Increment
            
#         else:
#                 Increment = np.zeros(NA)
        
#     else:
#         Increment = np.zeros(NA)
        
# else:
#     simu.time_since_loss[it] = 0
#     Increment = np.zeros(NA)
        
# Increment = config.FT['search']*Increment

# usearch = simu.SearchCommand[it] + Increment
# # The command is sent at the next time, that's why we note it+1
# simu.SearchCommand[it+1] = usearch


""" Implementation RELOCK incremental 11/05/2022"""
""" usearch est maintenant un delta à ajouter à la position actuelle des LAR """

# IgdRank = np.linalg.matrix_rank(simu.Igd[it])
# NotCophased = (IgdRank < NA-1)
# simu.IgdRank[it] = IgdRank

# if NotCophased:
#     simu.time_since_loss[it]=simu.time_since_loss[it-1]+config.dt
    
#     # FringeLost = (NotCophased and (IgdRank<np.linalg.matrix_rank(simu.Igd[it-1]))
#     # This situation could pose a problem but we don't manage it yet        
#     if (simu.time_since_loss[it] > config.FT['SMdelay']):
        
#         Igdna = np.dot(config.FS['OPD2Piston'],
#                        np.dot(simu.Igd[it],config.FS['Piston2OPD']))
        
#         CophasedBaselines=np.where(np.diag(simu.Igd[it])>0.5)[0]
#         CophasedPairs=[]
#         for ib in CophasedBaselines:
#             ia,iap = config.FS['ich'][ib][0], config.FS['ich'][ib][1]
#             CophasedPairs.append([ia,iap])
            
#         CophasedGroups = JoinOnCommonElements(CophasedPairs)
        
#         # Fringe loss
#         simu.LostTelescopes[it] = (np.diag(Igdna) == 0)*1      # The positions of the lost telescopes get 1.
#         # WeLostANewTelescope = (sum(newLostTelescopes) > 0)
        
#         # Photometry loss
#         simu.noSignal_on_T[it] = 1*(simu.SNRPhotometry[it] < config.FT['ThresholdPhot'])
            
#         comparison = (simu.noSignal_on_T[it] == simu.LostTelescopes[it])
#         simu.LossDueToInjection[it] = (comparison.all() and sum(simu.noSignal_on_T[it])>1)       # Evaluates if the two arrays are the same
        
#         if not simu.LossDueToInjection[it]:     # The fringe loss is not due to an injection loss
#             ### On entre dans le mode RELOCK
            
#             config.FT['state'][it] = 1          # Variable de suivi d'état du FT

#             ### On regarde si de nouveaux télescopes viennent juste d'être perdus.
#             newLostTelescopes = (simu.LostTelescopes[it] - simu.LostTelescopes[it-1] == 1)
#             TelescopesThatGotBackPhotometry = (simu.noSignal_on_T[it-1] - simu.noSignal_on_T[it] == 1)
#             # WeGotBackPhotometry = (sum(TelescopesThatGotBackPhotometry) > 0)
            
#             TelescopesThatNeedARestart = np.argwhere(newLostTelescopes + TelescopesThatGotBackPhotometry > 0)
#             print(TelescopesThatNeedARestart)
        
#             ### Pour chaque télescope nouvellement perdu, on réinitialise la fonction usaw
#             for ia in range(NA):
#                 if (ia in TelescopesThatNeedARestart) or (config.FT['state'][it-1]!=1):
#                     config.FT['it0'] = it; config.FT['it_last'][ia]=it;
#                     config.FT['eps'] = 1
#                     config.FT['LastPosition'][ia] = 0

#             usaw,change = searchfunction_inc_basical(it)

#             #config.FT['usaw'][it]= usaw

#             Kernel = np.identity(NA) - Igdna
#             usearch = np.dot(Kernel,usaw*config.FT['Velocities'])
            

#             #usearch = usearch/np.ptp(usearch) * config.FT['maxVelocity']
            
#             # if change:  # Change direction of scan
#             #     # Fais en sorte que les sauts de pistons de télescopes cophasés 
#             #     # entre eux maintiennent l'OPD constante: si 1 et 2 sont cophasés
#             #     # avec OPD=p2-p1, au prochain saut le télescope 2 va à la position
#             #     # du T1 + OPD et pas à la position qu'il avait avant le précédent saut.
#             #     for group in CophasedGroups:    
#             #         for ig in range(1,len(group)):
#             #             ia = int(float(group[ig])-1) ; i0 = int(float(group[0])-1)
#             #             LastPosition[ia] = LastPosition[i0] + simu.SearchCommand[it,ia]-simu.SearchCommand[it,i0]
#             #     usearch = LastPosition + Increment
#             #     LastPosition = simu.SearchCommand[it]
                
#             # else:
#             #     usearch = simu.SearchCommand[it]+Increment
                
#             # config.FT['LastPosition'][it+1] = LastPosition
            
#             # You should send command only on telescope with flux
#             simu.NoPhotometryFiltration[it] = np.identity(NA) - np.diag(simu.noSignal_on_T[it])
#             usearch = np.dot(simu.NoPhotometryFiltration[it],usearch)
        
        
#         else:
#             usearch = 0#simu.SearchCommand[it]
    
#     else:
#         usearch = 0#simu.SearchCommand[it]
        
# else:
#     simu.time_since_loss[it] = 0
#     usearch = 0#simu.SearchCommand[it]
    
    
# usearch = config.FT['search']*usearch
# # The command is sent at the next time, that's why we note it+1
# simu.SearchCommand[it+1] = simu.SearchCommand[it]+usearch

