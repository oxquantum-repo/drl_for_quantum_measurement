# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:00:13 2019

@author: Vu
"""

import sys
import numpy as np
sys.path.append('../../')
sys.path.append('../../environments')
sys.path.append('/home/sebastian/PycharmProjects/Vu/Vu_algorithm/DPhil/release_drl_quantum_env/environments')

from quan_T4_2d import Quantum_T4_2D
#from quan_env_rand_loc_T4_2d_norepeat import Quantum_T4_2D_Norepeat
#from quan_basel2_2d_norepeat import Quantum_Basel2_2D_Norepeat
#from quan_basel2_2d_small import Quantum_Basel2_2D_Small
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

def plot_arrow_action(myactions,ax):
    width = 1
    headwidth = 6.
    frac = 0.3
    shrink = 0.05
    linewidth = 1
    color = 'red'
    
    
    for ii in range(len(myactions)):
        action=[myactions[ii]]
        
        if 5 in action:
        # Left Down
            ax.annotate("", xy=(0.1,0.1), xytext=(0.9, 0.9), arrowprops=dict(width=width,
        headwidth = headwidth,frac = frac,shrink = shrink,linewidth =linewidth,color = color))
        if 4 in action:
    
            # Right Top
            ax.annotate("", xy=(0.9,0.9), xytext=(0.1, 0.1), arrowprops=dict(width=width,
        headwidth = headwidth,frac = frac,shrink = shrink,linewidth =linewidth,color = color))
        
        if 1 in action:
        # Down
        
            ax.annotate("", xy=(0.5,0.1), xytext=(0.5, 0.9), arrowprops=dict(width=width,
        headwidth = headwidth,frac = frac,shrink = shrink,linewidth =linewidth,color = color))
        
        
        
        # Left
        if 2 in action:
            ax.annotate("", xy=(0.1,0.5), xytext=(0.9, 0.5), arrowprops=dict(width=width,
        headwidth = headwidth,frac = frac,shrink = shrink,linewidth =linewidth,color = color))
        
        
        # Right
        if 3 in action:
            ax.annotate("", xy=(0.9,0.5), xytext=(0.1, 0.5), arrowprops=dict(width=width,
        headwidth = headwidth,frac = frac,shrink = shrink,linewidth =linewidth,color = color))
         
        # Up
        if 0 in action:
            ax.annotate("", xy=(0.5,0.9), xytext=(0.5, 0.1), arrowprops=dict(width=width,
        headwidth = headwidth,frac = frac,shrink = shrink,linewidth =linewidth,color = color))

        

    return ax

def plot_arrow_to_file(newenv,optimal_policy_list,optimal_val_list,optimal_policy_list_2
                       ,optimal_val_list_2,strFolder,myxlabel=None,myxrange=None,
                       myylabel=None,myyrange=None):
    
    for tt,optimal_policy in tqdm(enumerate(optimal_policy_list)):
        #optimal_policy=optimal_policy_list[-1]
        if tt<0:
            continue
        #f, axarr  = plt.subplots(newenv.dim[0],newenv.dim[1],figsize=(16,16))
        f, axarr  = plt.subplots(newenv.dim[0],newenv.dim[1],figsize=(30,26))
        myax=f.add_subplot(111, frameon=False)
     
        #plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel(myxlabel,fontsize=50)
        plt.ylabel(myylabel,fontsize=50)
        myxtick=np.linspace(1,newenv.dim[0]-2,4)
        #print(myxtick)
        myax.set_xticks(myxtick)
        myytick=np.linspace(1,newenv.dim[1]-2,4)
        #print(myytick)
        myax.set_yticks(myytick)
        myax.set_xticklabels(myxrange,fontsize=44)
        myax.set_yticklabels(myyrange,fontsize=44)
        
        for ii in range(newenv.dim[0]): # all rows
            for jj in range(newenv.dim[1]): # all columns
        
                #action=np.random.randint(0,5)
                val=optimal_val_list[tt][ii,jj]
                val2=optimal_val_list_2[tt][ii,jj]
                if np.abs(val-val2)<0.1:
                    actions=[optimal_policy[ii,jj], optimal_policy_list_2[tt][ii,jj]]
                else:
                    actions=[optimal_policy[ii,jj]]
                    
                axarr[ii,jj]=plot_arrow_action(actions,axarr[ii,jj])
                
                #print(myxlabel,myxrange,myylabel,myyrange)
                #axarr[ii,jj].set_xlabel(myxlabel,fontsize=14)
                #axarr[ii,jj].set_ylabel(myylabel,fontsize=14)
                #axarr[ii,jj].set_xticklabels(myxrange)
                #axarr[ii,jj].set_yticklabels(myyrange)
        
                #mydist_str="{:.1f},{:.0f}".format(10*mydata[0],10*mydata[1])
                #axarr[ii,jj].set_title(mydist_str)
                #print(ii,jj,mydata)
                        
                axarr[ii,jj].axis('off')                      
        

        strFile="{:s}/policy_{}.pdf".format(strFolder,tt)
        f.tight_layout()
        f.savefig(strFile,boxes_inches="tight") 
        f.clf()
        plt.close()
        gc.collect()