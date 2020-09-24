import sys

sys.path.append('../')
sys.path.append('../../')

sys.path.append('../../environments')
sys.path.append('../utilities')
sys.path.append('/home/sebastian/PycharmProjects/Vu/Vu_algorithm/DPhil/release_drl_quantum_env/environments')
sys.path.append('/home/sebastian/PycharmProjects/Vu/Vu_algorithm/DPhil/release_drl_quantum_env/utilities')

import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import gc      


def print_trajectory_from_location(env,loc_state_list, idx,myxlabel,myxrange
                                       ,myylabel,myyrange, strFolder="",filetype="pdf"):
    #width,height=IM_SIZE,IM_SIZE

    #imgheight,imgwidth=512,512
    
    n=env.dim[0]
    m=env.dim[1]
    
    mask=np.zeros((n,m)) # 1 : taken
        
    for c in range(len(loc_state_list)):    
        f, axarr = plt.subplots(n,m,figsize=(12,12))
        
        myax=f.add_subplot(111, frameon=False)
     
        #plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel(myxlabel,fontsize=24)
        plt.ylabel(myylabel,fontsize=24)
        myxtick=np.linspace(1,env.dim[0]-2,4)
        #print(myxtick)
        myax.set_xticks(myxtick)
        myytick=np.linspace(1,env.dim[1]-2,4)
        #print(myytick)
        myax.set_yticks(myytick)
        myax.set_xticklabels(myxrange,fontsize=24)
        myax.set_yticklabels(myyrange,fontsize=24)
        
        
        for ii in range(c):

            row,col=loc_state_list[ii]
            
            mask[row,col]=1

            block_data=np.reshape(env.data[row][col],(3**len(env.dim),2))
            
            #patch=env.data[row][col]
            idxQ=0
            for uu in range(9):
                isQuantum,scoreQ=env.predict_scoreQuantum(block_data[uu])         
                if isQuantum is True:
                    idxQ=uu
                                   
            patch=env.image_smallpatch_data[row][col][idxQ]
            plt.rcParams['image.cmap'] = 'viridis'

            axarr[row,col].imshow(patch,vmin=0,vmax=0.7)
            axarr[row,col].axis('off')

            # Create a Rectangle patch
			#rect = patches.Rectangle((patch.shape[0],patch.shape[1]),40,30,linewidth=1,edgecolor='r',facecolor='none')

			# Add the patch to the Axes
			#axarr[row,col].add_patch(rect)
    
            # plot the rest of the images with empty
            for iRow in range(n):
                for iCol in range(m):
                
                    if mask[iRow,iCol]==0:
                        
                        cmap = plt.cm.OrRd
                        cmap.set_bad(color='black')

                        patch=np.zeros_like(env.image_smallpatch_data[0][0][4])
                        
                        axarr[iRow,iCol].imshow(patch,vmin=0,vmax=0.7,cmap=cmap)
                        axarr[iRow,iCol].axis('off')

        if filetype=="pdf":
            strPath=strFolder+"path_{}_{}.pdf".format(idx,c)
        else:
            strPath=strFolder+"path_{}_{}.png".format(idx,c)
            
        print(strPath)

        sleep(0.02)
        f.savefig(strPath,bbox_inches = 'tight')
        
        # release RAM
        f.clf()
        plt.close()
        gc.collect()
        
        
        
def print_policy_map(env,loc_state_list, idx):

    n=env.dim[0]
    m=env.dim[1]
    
    mask=np.zeros((n,m)) # 1 : taken
        
    for c in range(len(loc_state_list)):    
        f, axarr = plt.subplots(n,m,figsize=(12,12))
        for ii in range(c):

            row,col=loc_state_list[ii]
            mask[row,col]=1
            
            patch=env.data[row][col]
    
            axarr[row,col].imshow(patch)
            axarr[row,col].axis('off')
    
            # plot the rest of the images with empty
            for iRow in range(n):
                for iCol in range(m):
                
                    if mask[iRow,iCol]==0:
                        patch=np.zeros_like(env.data[0][0])
                        
                        axarr[iRow,iCol].imshow(patch)
                        axarr[iRow,iCol].axis('off')


        strPath="plot/b2/path_{}_{}.png".format(idx,c)
        print(strPath)
        f.savefig(strPath,bbox_inches = 'tight')
        
        
        
def final_policy_on_test(newenv,model,starting_loc):

    #row_pixel_idx,col_pixel_idx=[np.random.randint(0,env.imgheight),np.random.randint(0,env.imgwidth)]
    #starting_loc=[row_pixel_idx,col_pixel_idx]
    #print("starting_loc",starting_loc)

    nrow,ncol=newenv.dim
    optimal_policy=np.zeros((nrow,ncol))
    optimal_policy_2=np.zeros((nrow,ncol))
    val_policy=np.zeros((nrow,ncol))
    val_policy_2=np.zeros((nrow,ncol))

    for ii in range(nrow):
        for jj in range(ncol):
            loc_state=[ii,jj]
            #print(loc_state,nrow,ncol)
            state=newenv.get_state(loc_state)

            action,val,action_2,val_2=model.sample_action(newenv,state,loc_state,eps=0,isNoOverlapping=False,is2Action=True)
            #print(action)
            optimal_policy[ii,jj]=action
            optimal_policy_2[ii,jj]=action_2
            val_policy[ii,jj]=val
            val_policy_2[ii,jj]=val_2


    return optimal_policy,val_policy,optimal_policy_2,val_policy_2

def get_value_state_on_test(model,newenv):

    #row_pixel_idx,col_pixel_idx=[np.random.randint(0,env.imgheight),np.random.randint(0,env.imgwidth)]
    #starting_loc=[row_pixel_idx,col_pixel_idx]
    #print("starting_loc",starting_loc)

    nrow,ncol=newenv.dim
    value_state_map=np.zeros((nrow,ncol))
    for ii in range(nrow):
        for jj in range(ncol):
            loc_state=[ii,jj]
            #print(loc_state,nrow,ncol)
            state=newenv.get_state(loc_state)

            neighborMaps=newenv.get_neighborMap(loc_state)

            value_state=model.get_value_state(state,neighborMaps)
            #print(action)
            temp="{0:.2f}".format(np.float(value_state))
            value_state_map[ii,jj]=temp

    return value_state_map

def final_policy(env,model):

    env.reset()
    nrow,ncol=env.dim
    optimal_policy=np.zeros((nrow,ncol))
    for ii in range(nrow):
        for jj in range(ncol):
            loc_state=[ii,jj]
            state=env.get_state(loc_state)

            action=model.sample_action(env,state,loc_state,eps=0)
            optimal_policy[ii,jj]=action
    return optimal_policy
    