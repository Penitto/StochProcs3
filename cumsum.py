#Create the change point detection method.
from scipy import stats
import numpy as np
def change_point_detection(df_column,length_of_interval=20,specified_confidence_level=0.95,niterations=100,acc=False,rank=False):
    """
    Uses the change point algorithm to find change points in the time series data. More information plus the CUMSUM algorithm can be found in the article linked here:
        https://variation.com/wp-content/uploads/change-point-analyzer/change-point-analysis-a-powerful-new-tool-for-detecting-changes.pdf
        https://variation.com/change-point-analysis-a-powerful-new-tool-for-detecting-changes/
	
    Args:
	*df_column: list(float). A single list of numeric time series data points.
	*length_of_interval: even int. the length of the interval to search for a change point. If a change point is not found within the interval, then the algorithm searches again, this time including half the original interval. For example, if the length_of_interval=20 and len(df_column)=50, assuming there is no change point, the algorithm will search intervals 0-19, 10-29, 20-39, 30-49. If there are change points detected at points 8 and 32, the algorithm will search 0-20, 8-27, 18-37, 32-49. (default 20)
	*specified_confidence_level: float (0.0,1.0). The confidence level which the change point is detected. (default 0.95)
	*niterations: int > 10. The number of bootstrapping iterations in the CUMSUM algorithms (default 100)
	*acc: boolean. If True, df_column time series points will be subtacted from their previous points, obtaining the 'velocity' of the time series. The change point detection algorithm is used on the 'velocity'. (default False)
	*rank: boolean. If True, time series points within the change point analysis are ranked by value and the algorithm is used on the ranked time series data. (default False)
	
    Returns:
	*change_point_indices: list(int). returns a list of indices where change points are found. The last index and first index are always included as change points. 
    """
    # change_point_detection function's assertions
    # assert that the df_column is a list, the length is greater than 4
    assert(isinstance(df_column,list) and len(df_column)>4 )
    # assert all values in df_column are integers or floats and not nan nor infinities
    assert(np.sum([not (np.isreal(x) and np.isfinite(x)) for x in df_column])==0)
    # assert the length_of_interval is greater than 0 and is an integer
    assert(isinstance(length_of_interval,int) and length_of_interval>=4 and length_of_interval%2==0)
    # assert the specified_confidence_level is between 0 and 1
    assert(specified_confidence_level<1 and specified_confidence_level>0)
    # assert niterations is greater than or equal to 10 and is an integer
    assert(isinstance(niterations,int) and niterations>=10)
    # assert acc is boolean
    assert(isinstance(acc,bool))
    # assert rank is boolean
    assert(isinstance(rank,bool))
    #Starts the time at the length_of_interval
    time = length_of_interval
    #Creates a list of change point indices
    change_point_indices = []
    change_point_indices.append(0)
    while time<len(df_column):
        change_point_index = change_points(df_column[time-length_of_interval:min(time,len(df_column))],specified_confidence_level,niterations,acc,rank)
        #If a change point index is found, add the time-length_of_interval to it.
        #For example, a changepoint at 26 found in the interval 20:30 will give a index value of 6. add 20.
        if change_point_index:
            change_point_index+=(time-length_of_interval)
        if not change_point_index:
            #if no change_point was produced, look from half the time.
            #For example, if no change_point was produced in the interval 30:40, look in the interval 35:45.
            #This makes sure no boundary points go unnoticed.
            change_point_index = change_points(df_column[int(min(time-length_of_interval/2,len(df_column)-1)):int(min(time+length_of_interval/2,len(df_column)))],specified_confidence_level,niterations,acc,rank)
            if change_point_index:
                #If one is found, add onto it the min(time-length_of_interval/2,len(df_column)-1)
                change_point_index+=int(min(time-length_of_interval/2,len(df_column)-1))
        if change_point_index:
            #If there is a change_point_index, append it to the list and make the time equal to the index+1
            change_point_indices.append(int(change_point_index))
            time = int(change_point_index)+length_of_interval+1
        if not change_point_index:
            #If there is still no change_point_index, add the length of the interval to the time
            time+=length_of_interval
    change_point_indices.append(len(df_column))
    return change_point_indices
    
def change_points(column,specified_confidence_level=0.95,niterations=100,acc=False,rank=False):
    """
    The CUMSUM change point algorithm. The CUMSUM algorithm can be found in the article linked here:
        https://variation.com/wp-content/uploads/change-point-analyzer/change-point-analysis-a-powerful-new-tool-for-detecting-changes.pdf
	https://variation.com/change-point-analysis-a-powerful-new-tool-for-detecting-changes/
	
    Args:
	*column: list(float). A single list of numeric time series data points.
	*specified_confidence_level: float (0.0,1.0). The confidence level which the change point is detected. (default 0.95)
	*niterations: int > 10. The number of bootstrapping iterations in the CUMSUM algorithms (default 100)
	*acc: boolean. If True, df_column time series points will be subtacted from their previous points, obtaining the 'velocity' of the time series. The change point detection algorithm is used on the 'velocity'. (default False)
	*rank: boolean. If True, time series points within the change point analysis are ranked by value and the algorithm is used on the ranked time series data. (default False)
	
    Returns:
	*change_point_index: int/None. if there is a change point, it returns the index where the change point is expected to be located. If a change point is not found, then None is returned. 
    """
    # change_points function's assertions
    # assert that the column is a list, the length is greater than 4
    assert(isinstance(column,list) and len(column)>4 )
    # assert all values in column are integers or floats and not nan nor infinities
    assert(np.sum([not (np.isreal(x) and np.isfinite(x)) for x in column])==0)
    # assert the specified_confidence_level is between 0 and 1
    assert(specified_confidence_level<1 and specified_confidence_level>0)
    # assert niterations is greater than or equal to 10 and is an integer
    assert(isinstance(niterations,int) and niterations>=10)
    # assert acc is boolean
    assert(isinstance(acc,bool))
    # assert rank is boolean
    assert(isinstance(rank,bool))
    if acc:
        #If acceleration, then find the differences first. 
        new_column =[0]
        for i in range(1,len(column)):
            new_column.append(column[i]-column[i-1])
        #Make the first entry equal the average of the next two entries
        new_column[0] = np.mean(new_column[1:3])
        column = new_column
    if rank:
        #If ranked, then rank the numbers in the column from least(1) to greatest(len(column)).
        rank_column = stats.rankdata(column)
        column = rank_column.tolist()
	#Find the average
        column_mean = np.mean(column)
	#create the cumulative sum list
        cum_sum_list = [0]
        for i in range(0,len(column)):
            cum_sum_list.append(cum_sum_list[i]+column[i]-column_mean)
	#A magnitude of the necessary change is required. This will be the difference 
	#between the maximum and minimum values of the cumulative sum list
        s_diff = max(cum_sum_list)-min(cum_sum_list)
	#Bootstrapping now occurs. A set number of iterations (n) will occur. Reorder
	#the points, calculate the cumulative sums, and find the difference of the max and min.
	#final output will be a list of n values stating the differences.
        list_of_diffs = []
        n = niterations
        for i in range(0,n):
            reordered_list = np.random.choice(column,len(column),replace=False)
            reordered_cum_sum_list = [0]
            for j in range(0,len(reordered_list)):
                reordered_cum_sum_list.append(reordered_cum_sum_list[j]+reordered_list[j]-column_mean)
            list_of_diffs.append(max(reordered_cum_sum_list)-min(reordered_cum_sum_list))
        confidence_level = np.mean([x<s_diff for x in list_of_diffs])
        change_point_index = np.argmax(map(abs,cum_sum_list)) if confidence_level>specified_confidence_level else None
        return change_point_index  
        
#Once the change point indices are found, use the average and column velocity to determine if there
#is an increasing or decreasing or constant velocity. 
def bit_depth_change(df_column,change_point_indices,p_threshhold=.05,avg_threshhold=0):
    """
    Once the change_point_indices are found using the change_point_detection function, this function determines whether the time series between change points is increasing, decreasing, or constant in value. 
	
    Args:
	*df_column: list(float). A single list of numeric time series data points. The same df_column used in the change_point_detection function.
	*change_point_indices: list(int). The change_point_indices returned from the change_point_detection function.
	*p_threshhold: float (0.0,1.0). The p-value determining whether the time series is increasing or decreasing. (default 0.05)
	*avg_threshhold: float > 0. If the statistic is less than the p_threshhold, then the interval is increasing or decreasing. The absolute average of the slope must be greater than the avg_threshhold to be increasing or decreasing. (default 0)
	
    Returns:
	*bit_depth_change: list(int). Returns a list the size of df_column that correlates with the slope observed at that location. If -1, the slope is decreasing. If 0, the slope is constant. If 1, the slope is increasing. 
    """
    # bit_depth_change function's assertions
    # assert that the df_column is a list, the length is greater than 4
    assert(isinstance(df_column,list) and len(df_column)>4 )
    # assert all values in df_column are integers or floats and not nan nor infinities
    assert(np.sum([not (np.isreal(x) and np.isfinite(x)) for x in df_column])==0)
    # assert that the change_point_indices is a list, the length is greater than or equal to 2
    assert(isinstance(change_point_indices,list) and len(change_point_indices)>=2 )
    # assert the p_threshhold is between 0 and 1
    assert(p_threshhold<1 and p_threshhold>0)
    # assert the avg_threshhold is greater than or equal to 0
    assert(avg_threshhold>=0)
    #Find the velocity between each second interval. Make the velocity at second 0 equal
    #to the average velocity of second 1 and 2.
    v_list = [df_column[index]-df_column[index-1] for index,item in enumerate(df_column)]
    v_list[0] = np.mean(v_list[1:3])
    bit_depth_change = [0]*len(df_column)
    for i in range(1,len(change_point_indices)):
        #Get the list of points used
        v_list_used = v_list[change_point_indices[i-1]:change_point_indices[i]]
        #Within the velocity list between each change point, remove the possible outliers
        #with the interquartile range method.
        iqr = np.subtract(*np.percentile(v_list_used,[75,25]))
        #Find the upper and lower bounds for the range.
        lower_bound = np.percentile(v_list_used, 25)-1.5*iqr
        upper_bound = np.percentile(v_list_used, 75)+1.5*iqr
        #Use the upper and lower bounds to remove possible outliers
        final_list = [x for x in v_list_used if x<=upper_bound and x>=lower_bound]
        #use the p value todecide whether the final_list has a mean value of 0.
        tstat, pvalue = stats.ttest_1samp(final_list,0,nan_policy='omit')
        #if the p-value is below the pthreshhold, then it is possibly changing velocity.
        if pvalue<p_threshhold:
            #The first test has showed signs of possible change. Now find if the average
            #change is large enough for it to be a change in BD
            if abs(np.mean(final_list))>avg_threshhold:
                #The second test has been passed. There is a change in depth during this interval.
                #Make all points within this interval reflect this change. The average will
                #tell whether it is a positive change or a negative change.
                if np.mean(final_list)>0:
                    bit_depth_change[change_point_indices[i-1]:change_point_indices[i]] = [1]*(change_point_indices[i]-change_point_indices[i-1])
                elif np.mean(final_list)<0:
                    bit_depth_change[change_point_indices[i-1]:change_point_indices[i]] = [-1]*(change_point_indices[i]-change_point_indices[i-1])
        #if there is 
    #return the column of bd direction
    return bit_depth_change            
    
#Smooth out the process of finding the bit depth change. If there is an interval 
#less that secs seconds, half goes to the previous interval and half goes to the latter.
def direction_smoother(direction_column,secs=0):
    """
    Once the bit_depth_change column is obtained from the bit_depth_change function, this function can be used to smooth any increasing, decreasing, or constant intervals too small in time. If there is an interval less that secs, half goes to the previous interval and half goes to the latter.
	
    Args:
	*direction_column: list(int). The bit_depth_change column obtained from the bit_depth_change function.
	*secs: int > 0. If there is an interval less that secs, half goes to the previous interval and half goes to the latter. (default 0)
	
    Returns:
	*direction_column: list(int). Returns a list the size of the provided direction_column that correlates with the slope observed at that location. If -1, the slope is decreasing. If 0, the slope is constant. If 1, the slope is increasing. Now there are no intervals less than secs. 
    """
    # direction_smoother function's assertions
    # assert that the direction_column is a list
    assert(isinstance(direction_column,list) )
    # assert all values in direction_column are -1, 0, or 1
    assert(np.sum([not (x in [-1,0,1]) for x in direction_column])==0)
    # assert secs is greater than or equal to 10 and is an integer
    assert(isinstance(secs,int) and secs>=0)
    time = 0
    while time<len(direction_column):
        for i in range(time+1,len(direction_column)):
            if direction_column[i]!=direction_column[time]:
                if i-time<secs and time==0:
                    None
                elif i-time<secs:
                    direction_column[time:time+(i-time)/2] = [direction_column[time-1]]*((i-time)/2)
                    direction_column[time+(i-time)/2:i] = [direction_column[i+1]]*(i-(time+(i-time)/2))
                    if direction_column[time-1] == direction_column[i+1]:
                        for j in range(i+1,len(direction_column)):
                            if direction_column[j]!=direction_column[i+1]:
                                i = j
                                break
                            elif j == len(direction_column)-1:
                                i = len(direction_column)
                                break
                time = i
                break
            elif i==len(direction_column)-1:
                if i-time<secs:
                    direction_column[time:len(direction_column)] = [direction_column[time-2]]*(len(direction_column)-time)
                time = len(direction_column)
                break
    return direction_column

#find if there are any change points within the change in direction_column. For example, find
#if there are any changes in successive 0s, then once that turns into 1 ( or -1)
#find if there is any change in that.
def change_point_loop(direction_column,df_column,scl=0.95,niter=100,acc=False,rank=False,p_thresh=0.05,avg_thresh=0):
    """
    Find if there are any change points within the change in direction_column. For example, find if there are any changes in successive 0s, then once that turns into 1 ( or -1) find if there is any change in that.
	
    Args:
	*direction_column: list(int). The bit_depth_change column obtained from the bit_depth_change function or the direction_smoother function.
	*df_column: list(float). The original df_column used in the change_point_detection function.
	*scl: float (0.0,1.0). (specified_confidence_level)The confidence level which the change point is detected. (default 0.95)
	*niter: int > 10. (niterations)The number of bootstrapping iterations in the CUMSUM algorithms (default 100)
	*acc: boolean. If True, df_column time series points will be subtacted from their previous points, obtaining the 'velocity' of the time series. The change point detection algorithm is used on the 'velocity'. (default False)
	*rank: boolean. If True, time series points within the change point analysis are ranked by value and the algorithm is used on the ranked time series data. (default False)
	*p_thresh: float (0.0,1.0). (p_threshhold)The p-value determining whether the time series is increasing or decreasing. (default 0.05)
	*avg_thresh: float > 0. (avg_threshhold)If the statistic is less than the p_threshhold, then the interval is increasing or decreasing. The absolute average of the slope must be greater than the avg_threshhold to be increasing or decreasing. (default 0)
	
    Returns:
	*direction_column: list(int). Returns a list the size of the provided direction_column that correlates with the slope observed at that location. If -1, the slope is decreasing. If 0, the slope is constant. If 1, the slope is increasing. 
    """
    # change_point_loop function's assertions
    # assert that the direction_column is a list
    assert(isinstance(direction_column,list) )
    # assert all values in direction_column are -1, 0, or 1
    assert(np.sum([not (x in [-1,0,1]) for x in direction_column])==0)
    # assert that the df_column is a list, the length is greater than 4
    assert(isinstance(df_column,list) and len(df_column)>4 )
    # assert all values in df_column are integers or floats and not nan nor infinities
    assert(np.sum([not (np.isreal(x) and np.isfinite(x)) for x in df_column])==0)
    # assert the scl is between 0 and 1
    assert(scl<1 and scl>0)
    # assert niter is greater than or equal to 10 and is an integer
    assert(isinstance(niter,int) and niter>=10)
    # assert acc is boolean
    assert(isinstance(acc,bool))
    # assert rank is boolean
    assert(isinstance(rank,bool))
    # assert the p_thresh is between 0 and 1
    assert(p_thresh<1 and p_thresh>0)
    # assert the avg_thresh is greater than or equal to 0
    assert(avg_thresh>=0)
    changes = 1
    loops = 0
    #Create a list of used change points. Loop will not test previously tested change points
    used_cp = []
    while changes > 0:
        loops+=1
        changes = 0
        time = 0
        while time<len(direction_column):
            #Get the initial state of the direction_column
            init_state = direction_column[time]
            #Find if and when the state of the direction_column changes.
            for i in range(time+1,len(direction_column)):
                if direction_column[i]!=direction_column[time]:
                    fin_state = i
                    break
                elif i==(len(direction_column)-1):
                    fin_state = i+1
                    break
            #Find if there is a change point within the interval
            change_point_index = change_points(column=df_column[time:fin_state],specified_confidence_level=scl,niterations=niter,acc=acc,rank=rank)
            #if there aren't 20 datapoints from beginning to change_point and change_point to end, omit.
            #if change_point_index:
                #if change_point_index<20 or abs(fin_state-time-change_point_index)<20:
                #    change_point_index = None
            if change_point_index:
                if (change_point_index+time) not in used_cp:
                    used_cp.append(change_point_index+time)
                    change_point_indices = []
                    change_point_indices.append(0)
                    change_point_indices.append(change_point_index)
                    change_point_indices.append(fin_state-time)
                    #if there is a new change point within the list, redo the bit depth change for that section
                    init_state = direction_column[time] 
                    direction_column[time:fin_state] = bit_depth_change(df_column[time:fin_state],change_point_indices,p_thresh,avg_thresh)
                    if any(np.array(direction_column[time:fin_state])!=init_state):
                        changes +=1
            time=fin_state
        print (loops, changes)
    return direction_column
    
#Get the final direction within each change point. 
def final_direction(direction_column,df_column,p_threshhold=0.05,avg_threshhold=0):
    """
    Make sure the directions are going in the proper position after the smoothing and the change_point_loop.
	
    Args:
	*direction_column: list(int). The bit_depth_change column obtained from the bit_depth_change function, direction_smoother function or the change_point_loop function.
	*df_column: list(float). The original df_column used in the change_point_detection function.
	*p_threshhold: float (0.0,1.0). The p-value determining whether the time series is increasing or decreasing. (default 0.05)
	*avg_threshhold: float > 0. If the statistic is less than the p_threshhold, then the interval is increasing or decreasing. The absolute average of the slope must be greater than the avg_threshhold to be increasing or decreasing. (default 0)
	
    Returns:
	*direction_column: list(int). Returns a list the size of the provided direction_column that correlates with the slope observed at that location. If -1, the slope is decreasing. If 0, the slope is constant. If 1, the slope is increasing. 
    """
    # final_direction function's assertions
    # assert that the direction_column is a list
    assert(isinstance(direction_column,list) )
    # assert all values in direction_column are -1, 0, or 1
    assert(np.sum([not (x in [-1,0,1]) for x in direction_column])==0)
    # assert that the df_column is a list, the length is greater than 4
    assert(isinstance(df_column,list) and len(df_column)>4 )
    # assert all values in df_column are integers or floats and not nan nor infinities
    assert(np.sum([not (np.isreal(x) and np.isfinite(x)) for x in df_column])==0)
    # assert the p_threshhold is between 0 and 1
    assert(p_threshhold<1 and p_threshhold>0)
    # assert the avg_threshhold is greater than or equal to 0
    assert(avg_threshhold>=0)
	#Find all the change points
    change_point_indices = []
    change_point_indices.append(0)
    for i in range(1,len(direction_column)):
        if direction_column[i]!=direction_column[i-1]:
            change_point_indices.append(i)
    change_point_indices.append(len(direction_column))
    direction_column = bit_depth_change(df_column,change_point_indices,p_threshhold=p_threshhold,avg_threshhold=avg_threshhold)
    return direction_column