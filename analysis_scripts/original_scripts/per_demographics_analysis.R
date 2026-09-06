########################################
# Thomas Denby - 2/22/2019
# Rational Inference in Phonotactic Adaptation
# osf.io/a6pjv/
# 
# this analyzes results and demographic info from both experiments in 5 parts: 
# 1 - preps data
# 2 - basic counting stats for language background/impairment
# 3 - means for legality advantage by lang background
# 4 - linear regressions for legality advantage by lang background
# 
# note that the working directory should be set to the folder including this script
# this can be done below manually, or automatically (depending on if you're using R studio or sourcing the script)
########################################


library(plyr)
library(dplyr)
library(ggplot2)
library(stringr)
library(lme4)
library(boot)
library(scales)
library(rstudioapi)

## set directory to local folder, either in R studio, R, or manually ##
#R studio
setwd(dirname(getActiveDocumentContext()$path))

#sourced script
# setwd(getSrcDirectory()[1])
# dirname(sys.frame(1)$ofile)

#manually (fill in the path to the folder including this script)
# setwd('local_path_here/results_scripts')


#load analysis functions
source("analysis_functions.R")

#function that extracts numbers
get_numbers <- function(num_vector, age_or_length = 'length'){
  for (i in 1:length(num_vector)){
    x = num_vector[i]
    x = as.character(x)
    #pull first number in string
    x = regmatches(x, regexpr("[[:digit:]]+", x))
    
    #if no number, set to 0 for length of speaking, or 100 (i.e., never) for age of acquisition
    if (identical(x, character(0))){
      if (age_or_length == 'age'){
        x = 100
      }else{
        x = 0
      }
    }
    #update vector
    num_vector[i] <-  as.numeric(x)
  }
  return(num_vector)
}


#### 1 - data prep ####
#open data set and demographics info
per2_file_path <- 'data/per2_filteredSubjects.csv'
per2_data <- open_file(per2_file_path, experiment = 2)

per2_demo <- read.csv('data/per2_demographics_anon.csv')

per1_file_path <- 'data/per1_filteredSubjects.csv'
per1_data <- open_file(per1_file_path, experiment = 1)

per1_demo <- read.csv('data/per1_demographics_anon.csv')

#remove/rename mismatching columns to ensure per1 and per2 data structure identical
# colnames(per1_data)[colnames(per1_data)=="xp_num"] <- "condition_num"
per1_data <- subset(per1_data, select = -c(gender_cross))
per2_data <- subset(per2_data, select = -c(list_n))

#add column for xp_num and change per2's condition num
per1_data$xp_num <- 1
per2_data$xp_num <- 2
per2_data$condition_num = per2_data$condition_num + 7

all_demo <- rbind(per1_demo, per2_demo)
all_data <- rbind(per1_data, per2_data)
all_data <- all_data[all_data$gen == 0.5,]

#get second, third, and combined Ln experience in years (and combined in log years)
# all_demo$secondLanguageLength <- as.character(all_demo$secondLanguageLength)
all_demo$secondLanguageLength <- as.character(all_demo$secondLanguageLength)
all_demo$secondLanguageLength_num <- get_numbers(all_demo$secondLanguageLength)
all_demo$secondLanguageLength_num <- as.integer(all_demo$secondLanguageLength_num)

all_demo$thirdLanguageLength <- as.character(all_demo$thirdLanguageLength)
all_demo$thirdLanguageLength_num <- get_numbers(all_demo$thirdLanguageLength)
all_demo$thirdLanguageLength_num <- as.integer(all_demo$thirdLanguageLength_num)

all_demo$allLangLength <- all_demo$secondLanguageLength_num + all_demo$thirdLanguageLength_num
all_demo$allLangLength_log <- log(all_demo$allLangLength+1)

#get second and third L age of acquisition (set number to 100, rather than 0, if there's no Ln)
all_demo$secondLanguageAge <- as.character(all_demo$secondLanguageAge)
all_demo$secondLanguageAge_num <- get_numbers(all_demo$secondLanguageAge, "age")
all_demo$secondLanguageAge_num <- as.integer(all_demo$secondLanguageAge_num)

all_demo$thirdLanguageAge <- as.character(all_demo$thirdLanguageAge)
all_demo$thirdLanguageAge_num <- get_numbers(all_demo$thirdLanguageAge, "age")
all_demo$thirdLanguageAge_num <- as.integer(all_demo$thirdLanguageAge_num)


#add column for anyL2 experience and early L2 (acquisition younger than 6)

NN_langs <- c('french','german','italian','russian','spanish','swedish','vietnamese') #this is a hand-coded list of langauges based on L2 participants who input funky stuff and otherwise wouldn't be caught
all_demo$secondLanguage <- tolower(all_demo$secondLanguage)

all_demo$anyL2 <- ifelse(all_demo$allLangLength > 0,1,
                         ifelse(all_demo$secondLanguage %in% NN_langs, 1, 0))


all_demo$birth <- grepl("birth",all_demo$secondLanguageAge, ignore.case = T)

all_demo$earlyL2 <- ifelse(all_demo$secondLanguageAge_num < 6, 1,
                           ifelse(all_demo$thirdLanguageAge_num < 6, 1,
                                  ifelse(all_demo$birth == T, 1, 0)))

all_demo$anyL3 <- ifelse(all_demo$thirdLanguageLength_num > 0,1,0)


#find funky cases where there's early L2 but no anyL2
all_demo$problem_cases <- ifelse(all_demo$anyL2 == 0 & all_demo$earlyL2 == 1, T,F)
#for people who put odd stuff in # of years column, set to 0; set anyL2 to 1 for actual L2 speakers leftover
all_demo[all_demo$problem_cases == T,]$earlyL2 <-  ifelse(all_demo[all_demo$problem_cases == T,]$secondLanguage  %in% c("n/a",'na','0','none',NA,'no','only english','english','i don\'t know'),0,1)
all_demo[all_demo$problem_cases == T & all_demo$earlyL2 == 1,]$anyL2 <- 1

#add columns for L2, L3, or either of individual langauges
all_demo$L2_french <- grepl("french",all_demo$secondLanguage, ignore.case = T)
all_demo$L3_french <- grepl("french",all_demo$thirdLanguage, ignore.case = T)
all_demo$all_french <- ifelse(all_demo$L2_french == T, T, ifelse(all_demo$L3_french == T, T, F))

all_demo$L2_hungarian <- grepl("hungarian",all_demo$secondLanguage, ignore.case = T)
all_demo$L3_hungarian <- grepl("hungarian",all_demo$thirdLanguage, ignore.case = T)
all_demo$all_hungarian <- ifelse(all_demo$L2_hungarian == T, T, ifelse(all_demo$L3_hungarian == T, T, F))

all_demo$L2_hindi <- grepl("hindi",all_demo$secondLanguage, ignore.case = T)
all_demo$L3_hindi <- grepl("hindi",all_demo$thirdLanguage, ignore.case = T)
all_demo$all_hindi <- ifelse(all_demo$L2_hindi == T, T, ifelse(all_demo$L3_hindi == T, T, F))



# merge/prep data
colnames(all_demo)[names(all_demo) == "rand_ID"] <- "worker_ID"

passing_participants <- unique(all_data$worker_ID)
all_demo <- all_demo[all_demo$worker_ID %in% passing_participants,]
all_demo <- all_demo[!duplicated(all_demo$worker_ID),]

data_demo <- merge(all_data, all_demo, by = "worker_ID")

diff_conditions <- c(1,2,6,7,8,9)
shared_conditions <- c(3,4,5,10)
data_demo$diffVshared <- ifelse(data_demo$condition_num %in% diff_conditions, 1,0)

french_conditions <- c(1,2,4,5,6,7)
data_demo$french_speaker <- ifelse(data_demo$condition_num %in% french_conditions, 1,0)

xp_one <- filter(data_demo, xp_num == 1)
xp_1A <- filter(xp_one, condition_num < 5)
xp_1B <- filter(xp_one, condition_num > 4)
xp_two <- filter(data_demo, xp_num == 2)


#### 2 - demographic counting stats ####
#number of participants per experiment who were non-native (NN) speakers/had speech impairments
length(unique(xp_1A[xp_1A$dialect == 'NNS',]$worker_ID)) # 4 NN speakers in 1A
length(unique(xp_1B[xp_1B$dialect == 'NNS',]$worker_ID)) # 2 NN speakers in 1B
length(unique(xp_two[xp_two$dialect == 'NNS',]$worker_ID)) # 4 NN speakers in xp2

length(unique(xp_1A[xp_1A$dialect == 'OED',]$worker_ID)) # 2 non American English in 1A
length(unique(xp_1B[xp_1B$dialect == 'OED',]$worker_ID)) # 0 in 1B
length(unique(xp_two[xp_two$dialect == 'OED',]$worker_ID)) # 1 in 2

length(unique(xp_1A[xp_1A$impairmentRadios == 'yes',]$worker_ID)) # 2 speakers with impairments in 1A
length(unique(xp_1B[xp_1B$impairmentRadios == 'yes',]$worker_ID)) # 0 in 1B
length(unique(xp_two[xp_two$impairmentRadios == 'yes',]$worker_ID)) # 3 in 2


#number of participants who were french/hindi/hungarian speakers (for relevant experiment)

length(unique(xp_one[xp_one$all_french == T,]$worker_ID)) # 30 french speakers in xp1A and 1B

#0 hindi/hungarian speakers in xp2
length(unique(xp_two[xp_two$all_hindi == T,]$worker_ID)) 
length(unique(xp_two[xp_two$all_hungarian == T,]$worker_ID))

#n participants with Ln experience
length(unique(data_demo[data_demo$anyL2 == T,]$worker_ID)) # 216 participants with any L2 experience
length(unique(data_demo[data_demo$earlyL2 == T,]$worker_ID)) # 49 participants with early L2 experience
length(unique(data_demo[data_demo$anyL3 == T,]$worker_ID)) # 68 participants with any L3 experience


#create new simple data frame and merge it with legality effects for each subject
demo_simple <- data.frame(all_demo$worker_ID, all_demo$all_french, all_demo$anyL2,
                          all_demo$earlyL2, all_demo$anyL3, all_demo$allLangLength)
colnames(demo_simple) <- c("worker_ID","all_french","anyL2", "earlyL2", "anyL3", "allLangLength")

per1_diffs.genBySub <- read.csv("data/per1_subj_legality.csv")
per2_diffs.genBySub <- read.csv("data/per2_subj_legality.csv")

per2_diffs.genBySub$condition_num = per2_diffs.genBySub$condition_num + 7

diffs.genBySub <- rbind(per1_diffs.genBySub, per2_diffs.genBySub)

diffs.genBySub <- merge(diffs.genBySub,demo_simple, by = "worker_ID") #only keep the first match, since there are multiple rows for some workers in demo file


diffs.genBySub$diffVshared <- ifelse(diffs.genBySub$condition_num %in% diff_conditions, 1,0)

per1_diffs.genBySub <- filter(diffs.genBySub, condition_num < 8)

  
#### 3 - means ####

### legality advantage means ###
#early diffs
earlyL2_mean_diffs.genBySub <- ddply(diffs.genBySub,.(earlyL2),summarize,PropYes = mean(PropYes))

earlyL2_mean_diffs.genBySub$upperCI = -999
earlyL2_mean_diffs.genBySub$lowerCI = -999
for (i in 1:length(earlyL2_mean_diffs.genBySub$earlyL2)){
  boot.results <- boot(data = diffs.genBySub[(diffs.genBySub$earlyL2 == earlyL2_mean_diffs.genBySub$earlyL2[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  earlyL2_mean_diffs.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  earlyL2_mean_diffs.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

colnames(earlyL2_mean_diffs.genBySub) = c("earlyL2","legality_effect","upperCI","lowerCI")
earlyL2_mean_diffs.genBySub

#any diffs
anyL2_mean_diffs.genBySub <- ddply(diffs.genBySub,.(anyL2),summarize,PropYes = mean(PropYes))
anyL2_mean_diffs.genBySub$upperCI = -999
anyL2_mean_diffs.genBySub$lowerCI = -999
for (i in 1:length(anyL2_mean_diffs.genBySub$anyL2)){
  boot.results <- boot(data = diffs.genBySub[(diffs.genBySub$anyL2 == anyL2_mean_diffs.genBySub$anyL2[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  anyL2_mean_diffs.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  anyL2_mean_diffs.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

colnames(anyL2_mean_diffs.genBySub) = c("anyL2","legality_effect","upperCI","lowerCI")
anyL2_mean_diffs.genBySub

#french diffs
french_mean_diffs.genBySub <- ddply(per1_diffs.genBySub,.(all_french),summarize,PropYes = mean(PropYes))

french_mean_diffs.genBySub$upperCI = -999
french_mean_diffs.genBySub$lowerCI = -999
for (i in 1:length(french_mean_diffs.genBySub$all_french)){
  boot.results <- boot(data = diffs.genBySub[(diffs.genBySub$all_french == french_mean_diffs.genBySub$all_french[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  french_mean_diffs.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  french_mean_diffs.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

colnames(french_mean_diffs.genBySub) = c("all_french","legality_effect","upperCI","lowerCI")
french_mean_diffs.genBySub


### overall false recognition means ###
#early overall
ddply(diffs.genBySub,.(earlyL2),summarize,mean_yes=(mean(legal_prop)+mean(illegal_prop))/2)

all_genBySub <- ddply(data_demo[data_demo$gen == 0.5,],.(worker_ID,earlyL2, anyL2, all_french),summarize,PropYes = mean(yesResponse))

earlyL2_means.genBySub <- ddply(all_genBySub,.(earlyL2), summarize,meanPropYes = mean(PropYes))

# add in upper and lower bounds on 95% confidence interval
earlyL2_means.genBySub$upperCI = -999
earlyL2_means.genBySub$lowerCI = -999
for (i in 1:length(earlyL2_means.genBySub$earlyL2)){
  boot.results <- boot(data = all_genBySub[(all_genBySub$earlyL2 == earlyL2_means.genBySub$earlyL2[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  earlyL2_means.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  earlyL2_means.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

earlyL2_means.genBySub

#any overall 
anyL2_means.genBySub <- ddply(all_genBySub,.(anyL2), summarize,meanPropYes = mean(PropYes))

# add in upper and lower bounds on 95% confidence interval
anyL2_means.genBySub$upperCI = -999
anyL2_means.genBySub$lowerCI = -999
for (i in 1:length(anyL2_means.genBySub$anyL2)){
  boot.results <- boot(data = all_genBySub[(all_genBySub$anyL2 == anyL2_means.genBySub$anyL2[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  anyL2_means.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  anyL2_means.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

anyL2_means.genBySub

#french overall
all_french_means.genBySub <- ddply(all_genBySub,.(all_french), summarize,meanPropYes = mean(PropYes))

# add in upper and lower bounds on 95% confidence interval
all_french_means.genBySub$upperCI = -999
all_french_means.genBySub$lowerCI = -999
for (i in 1:length(all_french_means.genBySub$all_french)){
  boot.results <- boot(data = all_genBySub[(all_genBySub$all_french == all_french_means.genBySub$all_french[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  all_french_means.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  all_french_means.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

all_french_means.genBySub




#### 4 - models ####

## early L2 model - data from all experiments##
#format syllable without speaker column to get rid of speaker tags
data_demo[data_demo$condition_num < 8,]$syllable_no_speaker <- substring(data_demo[data_demo$condition_num < 8,]$syllable_no_speaker,4,6)

comp.glm_noblock <- glmer(response~legality*(diffVshared+earlyL2) +legality:earlyL2:diffVshared + 
                            (1 + legality|worker_ID) + (1 + legality|syllable_no_speaker)
                          ,data=data_demo,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(comp.glm_noblock) #no significant effects of earlyL2



## anyL2 model - data from all experiments##
comp.glm_anyL2 <- glmer(response~legality*(diffVshared+anyL2) +legality:anyL2:diffVshared + 
                          (1+legality|worker_ID) + (1+legality|syllable_no_speaker)
                        ,data=data_demo,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(comp.glm_anyL2) #no significant effects of anyL2

exp2_chi_sq_legality_main_effect <- capture.output(chiReport.func(anova(comp.glm_anyL2,update(comp.glm_anyL2,.~.-(Intercept)))))


## french model - data from Exp1 ##
comp.glm_french <- glmer(response~legality*(diffVshared+all_french) +legality:all_french:diffVshared + 
                          (1+legality|worker_ID) + (1+legality|syllable_no_speaker)
                        ,data=xp_one,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(comp.glm_french) # no significant effects of french
