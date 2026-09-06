############################################################
# Thomas Denby - 2/22/2019
# Rational Inferece in Phonotactic Learning
# osf.io/a6pjv/
#
# This is a script that provides various convenience functions.
# It is called from the analysis scripts, and is not meant to be used alone
############################################################

library(plyr)
library(dplyr)
library(ggplot2)
library(lme4)
library(boot)
library(scales)
library(stringr)


#lme convenience function
chiReport.func <- function(a){
  ifelse (a$"Pr(>Chisq)"[2] > .0001,return(paste("chisq(",a$"Chi Df"[2],")=",round(a$Chisq[2],2),", p = ",round(a	$"Pr(>Chisq)"[2],4),sep="")),return(paste("chisq(",a$"Chi Df"[2],")=",round(a$Chisq[2],2),", p < .0001")))
}

#function that opens results file
#pattern is the type of pattern being tested (i.e. variation/novar vs. type/token)
#for variation pattern, input "var", for type/token leave blank
open_file <- function(file_path, experiment){
  ## for testing ## 
  # file_path <- '/Users/tdenby/Box Sync/ORG-SCHOOL-WCAS-LINGUISTICS-SOUNDLAB/SoundlabUsers/tommy/dissertation/results/per2/per2_filteredSubjects.csv'
  # file_path <- 'data/per1_filteredSubjects.csv'
  # experiment <- 1
  
  #read data
  data = read.csv(file_path)
  
  #adds contrast coded familiarization vs. generalization column
  data$gen = ifelse(grepl('test',data$label),.5,-.5)
  
  #adds contrast coded legality column
  data$legality <- ifelse(grepl('illegal',data$label),-.5,.5)
  
  #adds dummy coded response column
  data$yesResponse = ifelse (data$response=="yes",1,0)
  
  # len <- length(unique(data$index))
  names(data)[names(data) == 'xp_num'] <- 'condition_num'
  
  if (experiment == 1){
    block_length <- 36
    gen_length <- 4
    num_blocks <- 13
    last_fam_block <- 4
    center_num <- 9 #centered around block 9 (i.e. centered for final 9 generalization blocks)
  } else{
    block_length <- 32
    gen_length <- 4
    num_blocks <- 10
    last_fam_block <- 2
    center_num <- 6.5 #centered around block 6.5 
  }

  
  #adds column with appropriate block numbers
  data$block <- 1
  
  #for blocks 1-4 (familiarization), 36 items/block; blocks 5-13 (generalization), 40 items/block
  for (i in 1:num_blocks-1){
    if (i < last_fam_block){
      data[data$index > (block_length*i - 1),]$block <- i+1
    }else{ 
      data[data$index > (block_length*i + gen_length*(i-last_fam_block) - 1),]$block <- i+1
    }
  }
  
  
  #adds block number column 
  data$block_centered <- data$block - center_num
  
  #make a column for syllable without speaker tag
  data$syllable <- as.character(data$syllable)
  data$syllable_no_speaker <- substring(data$syllable,nchar(data$syllable)-5,nchar(data$syllable))
  
  return(data)
}




# bootstrap estimates of 95% CI for proportion yes (across participants) in each condition
#bootstrap function; calculate mean within conditions
boot.mean.fnc <- function (data,indices,column_name = "PropYes"){
  # select data at each index
  d <- data[indices,]
  # calculate mean
  return(mean(d[,column_name]))
}


#bootstrap funtion that takes data and means table as arguments, returns means table with upper and lower CIs
get_boot_CIs <- function(df, means_table,column = "PropYes"){
  # add in upper and lower bounds on 95% confidence interval
  means_table$upperCI = -999
  means_table$lowerCI = -999
  boot.results <- boot(data = df,statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  means_table$upperCI = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  means_table$lowerCI = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
  return(means_table)
}


#takes data file and returns data frame with mean and 95% bootstrapped CIs
#for familiarization items
fam_mean_CI <- function(data){
  BySub <- ddply(data[data$gen ==-.5,],.(worker_ID),summarize,PropYes = mean(yesResponse)) %>% na.omit()
  means.BySub <- ddply(BySub,.(), summarize,meanPropYes = mean(PropYes))
  
  # add in upper and lower bounds on 95% confidence interval
  means.BySub$upperCI = -999
  means.BySub$lowerCI = -999
  
  for (i in 1:length(means.BySub$ExperimentLabels)){
    boot.results <- boot(data = BySub,statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
    means.BySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
    means.BySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
  }
  
  return(means.BySub)
}
