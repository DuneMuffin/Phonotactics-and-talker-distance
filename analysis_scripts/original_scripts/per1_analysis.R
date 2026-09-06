############################################################
# Thomas Denby - 2/22/2019
# Rational Inferece in Phonotactic Learning
# osf.io/a6pjv/
# 
# This is the analysis script for Experiments 1A and 1B.
# Sections include: 
#                  (1) data prep
#                  (2) means (familiarization, legality, legality by speaker, vowel, etc.)
#                  (3) Exp1A model
#                  (4) Exp1B model
# 
# Models are logistic mixed effects regressions. 
# 
# All the filepaths used to read and write files are relative 
# to the position of this script (so don't move it out of the folder!).
# 
# This means you'll have to set the working directory to the local folder that the script is in.
# 
# Doing so differs depending on how you're using the script:
#       if you're using RStudio, the script should work as is.
#       if you're sourcing the script, uncomment the appropriate section below (line 39).
#       alternatively, you can manually enter in the script's filepath on your computer (line 42)
############################################################


library(plyr)
library(dplyr)
library(lme4)
library(boot)
library(stringr)
library(rstudioapi)

## set directory to local folder, either in R studio, R, or manually ##
#R studio
setwd(dirname(getActiveDocumentContext()$path))

#sourced script
# setwd(getSrcDirectory()[1])

#manually (fill in the path to the folder including this script)
# setwd('local_path_here/results_scripts')

#load functions from script
source("analysis_functions.R")

################### 1 - data prep ################### 
#open file
per1_file_path <- 'data/per1_filteredSubjects.csv'
per1_data <- open_file(per1_file_path, experiment = 1)

#split into generalization (or gen) syllables vs. familiarization  (or fam) syllables
all_gen_syllables <- filter(per1_data, gen == 0.5)
all_fam_syllables <- filter(per1_data, gen == -0.5)

#format 'label' and 'syllable' columns for consistency, transparency
all_gen_syllables$label <- as.character(all_gen_syllables$label)
all_gen_syllables$label <- sub('french','FR',all_gen_syllables$label)
all_gen_syllables$label <- sub('_F','_F1',all_gen_syllables$label)
all_gen_syllables$label <- sub('_M','_M1',all_gen_syllables$label)
all_gen_syllables$label <- sub('_F1I','_F2',all_gen_syllables$label) #FR_F2 is the speaker added for Exp 1B
all_gen_syllables$label <- sub('NS','EN',all_gen_syllables$label) 

all_gen_syllables$syllable <- as.character(all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('french','FR',all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('_F','_F1',all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('_M','_M1',all_gen_syllables$syllable)
all_gen_syllables$syllable <- sub('_F1I','_F2',all_gen_syllables$syllable) #FR_F2 is the speaker added for Exp 1B
all_gen_syllables$syllable <- sub('NS','EN',all_gen_syllables$syllable) 

#add column of condition names matching condition numbers, add column for xp_num (conditions 1-4 in xp1A, conditions 5-7 in xp1B)
condition_names <- data.frame(c(1,2,3,4,5,6,7), c("weak-dif","strong-dif","native_shared","NN_shared","NN_shared2","strong-dif2","weak-dif2"))
colnames(condition_names) <- c("condition_num","condition")
all_gen_syllables <- merge(all_gen_syllables,condition_names)
all_gen_syllables$xp_num <- 0
all_gen_syllables$xp_num <- ifelse(all_gen_syllables$condition_num < 5,1,2)

#add column for speaker, gender, vowel, accent, and syllable (without speaker)
all_gen_syllables$speaker <- substring(all_gen_syllables$label,nchar(all_gen_syllables$label)-4,nchar(all_gen_syllables$label))
all_gen_syllables$gender <- substring(all_gen_syllables$label,nchar(all_gen_syllables$label)-1,nchar(all_gen_syllables$label)-1)
all_gen_syllables$vowel <- substring(all_gen_syllables$syllable, 8,8)
all_gen_syllables$accent <- substring(all_gen_syllables$speaker, 0,2)
all_gen_syllables$accent <- ifelse (all_gen_syllables$accent == "FR", "French", "English")
all_gen_syllables$syllable_no_speaker <- substring(all_gen_syllables$syllable, 7,9)

#turn response, workerID, legality, syllable etc. into factors
all_gen_syllables$response <- as.factor(all_gen_syllables$response)
all_gen_syllables$worker_ID <- as.factor(all_gen_syllables$worker_ID)
all_gen_syllables$syllable <- as.factor(all_gen_syllables$syllable)
all_gen_syllables$syllable_no_speaker <- as.factor(all_gen_syllables$syllable_no_speaker)


################### 2 - means ################### 
###false alarm rate on familiarization syllables for xp1A and xp1B###
exp1A_fam_means <- fam_mean_CI(all_fam_syllables[all_fam_syllables$condition_num < 5,])
exp1B_fam_means <- fam_mean_CI(all_fam_syllables[all_fam_syllables$condition_num > 4,])



### overall mean false alarm rate and mean legality advantage (FA on legal minus FA on illegal) 
### across participants (not including condition) for both xps ###
means.overall_FA_both_xp <- data.frame()
means.legality_both_xp <- data.frame()

for (i in unique(all_gen_syllables$xp_num)){
  relDat <- all_gen_syllables[all_gen_syllables$xp_num == i,]
  overall_FA_bySub <-  ddply(relDat,.(worker_ID),summarize,PropYes = mean(yesResponse))
  means.overall_FA <- ddply(overall_FA_bySub,.(), summarize,meanPropYes = mean(PropYes))
  
  #grab confidence intervals for overall false alarms for each experiment, add to table with both
  means.overall_FA <- get_boot_CIs(overall_FA_bySub, means.overall_FA)
  means.overall_FA$xp_num = i
  means.overall_FA_both_xp <- rbind(means.overall_FA_both_xp, means.overall_FA)
  
  # mean legality advantage  #
  differencesBySub <- ddply(relDat,.(worker_ID,condition),summarize,PropYes = mean(yesResponse[legality == 0.5]) - mean(yesResponse[legality == -0.5]))
  mean.differencesBySub <- ddply(differencesBySub,.(),summarize,PropYes = mean(PropYes))
  mean.differencesBySub <- get_boot_CIs(differencesBySub, mean.differencesBySub)
  mean.differencesBySub$xp_num = i
  
  means.legality_both_xp <- rbind(means.legality_both_xp, mean.differencesBySub)
}

means.overall_FA_both_xp
means.legality_both_xp

### FA rate on all gen syllables, broken down by condition ### 
all_genBySub <- ddply(all_gen_syllables,.(worker_ID,legality,condition_num,condition),summarize,PropYes = mean(yesResponse))

means.genBySub <- ddply(all_genBySub,.(condition_num, legality), summarize,meanPropYes = mean(PropYes))

# add in upper and lower bounds on 95% confidence interval
means.genBySub$upperCI = -999
means.genBySub$lowerCI = -999
for (i in 1:length(means.genBySub$legality)){
  boot.results <- boot(data = all_genBySub[(all_genBySub$legality == means.genBySub$legality[i]) & (all_genBySub$condition_num == means.genBySub$condition_num[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  means.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  means.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

means.genBySub


### mean legality advantage across conditions ###
legal_means <- filter(all_genBySub, legality == 0.5)
legal_means <- subset(legal_means, select = -2)
illegal_means <- filter(all_genBySub, legality == -0.5)
illegal_means <- subset(illegal_means, select = -2)

diffs.genBySub <- merge(legal_means,illegal_means, by = c("worker_ID", "condition_num", "condition"))
colnames(diffs.genBySub) = c("worker_ID", "condition_num","condition","legal_prop","illegal_prop")
diffs.genBySub$PropYes <- diffs.genBySub$legal_prop - diffs.genBySub$illegal_prop


mean_diffs.genBySub <- ddply(diffs.genBySub,.(condition_num,condition),summarize,PropYes = mean(PropYes))

# add in upper and lower bounds on 95% confidence interval
mean_diffs.genBySub$upperCI = -999
mean_diffs.genBySub$lowerCI = -999
for (i in 1:length(mean_diffs.genBySub$condition_num)){
  boot.results <- boot(data = diffs.genBySub[(diffs.genBySub$condition_num == mean_diffs.genBySub$condition_num[i]), ],statistic=boot.mean.fnc,R=1000) # 1000 bootstrap replicates
  mean_diffs.genBySub$upperCI[i] = boot.ci(boot.results,type="perc")$perc[,5] # get upper 95%ile
  mean_diffs.genBySub$lowerCI[i] = boot.ci(boot.results,type="perc")$perc[,4] # get lower 95%ile
}

colnames(mean_diffs.genBySub) = c("condition_num","condition","legality_effect","upperCI","lowerCI")


### means broken down by vowel, speaker, syllable, block, etc. ###

#means by vowel
vowel_all_genBySub <- ddply(all_gen_syllables,.(worker_ID,xp_num, condition,vowel,legality,accent),summarize,PropYes = mean(yesResponse)) %>% na.omit()
vowel_means.genBySub <- ddply(vowel_all_genBySub,.(xp_num,condition,vowel,legality), summarize,meanPropYes = mean(PropYes))
vowel_accent_condition_means<- ddply(vowel_all_genBySub,.(xp_num,condition,vowel,accent), summarize,meanPropYes = mean(PropYes))


#means by speaker
speaker_means<- ddply(all_gen_syllables,.(speaker,legality), summarize,meanPropYes = mean(yesResponse))

legal_speakers <- speaker_means[speaker_means$legality == 0.5,]
illegal_speakers <- speaker_means[speaker_means$legality == -0.5,]
speaker_differences <- legal_speakers$meanPropYes - illegal_speakers$meanPropYes

speaker_diff <- cbind(legal_speakers[-2],illegal_speakers$meanPropYes,speaker_differences)


#means by speaker + condition
speaker_means_condition<- ddply(all_gen_syllables,.(condition_num,condition_num,speaker,legality), summarize,meanPropYes = mean(yesResponse))

legal_speakers_condition <- speaker_means_condition[speaker_means_condition$legality == 0.5,]
illegal_speakers_condition <- speaker_means_condition[speaker_means_condition$legality == -0.5,]
speaker_differences_condition <- legal_speakers_condition$meanPropYes - illegal_speakers_condition$meanPropYes

speaker_diff_condition <- cbind(legal_speakers_condition[-3],illegal_speakers_condition$meanPropYes,speaker_differences_condition)


#write means
write.table(exp1A_fam_means, file = "means/per1A_experiment_fam_means.csv",sep = ",",row.names = FALSE)
write.table(exp1B_fam_means, file = "means/per1B_experiment_fam_means.csv",sep = ",",row.names = FALSE)
write.table(means.overall_FA_both_xp, file = "means/per1_experiment_overall_gen_means.csv",sep = ",",row.names = FALSE)
write.table(means.legality_both_xp, file = "means/per1_experiment_legality_means.csv",sep = ",",row.names = FALSE)
write.table(means.genBySub, file = "means/per1_condition_means.csv",sep = ",",row.names = FALSE)
write.table(diffs.genBySub, file = "data/per1_subj_legality.csv",sep = ",",row.names = FALSE)
write.table(mean_diffs.genBySub, file = "means/per1_condition_legality_means.csv",sep = ",",row.names = FALSE)
write.table(vowel_means.genBySub, file = "means/per1_vowel_condition_legality_means.csv",sep = ",",row.names = FALSE)
write.table(vowel_accent_condition_means, file = "means/per1_vowel_accent_condition_legality_means.csv",sep = ",",row.names = FALSE)
write.table(speaker_diff, file = "means/per1_speaker_legality_means.csv",sep = ",",row.names = FALSE)
write.table(speaker_diff_condition, file = "means/per1_speaker_condition_legality_means.csv",sep = ",",row.names = FALSE)






################### 3 - model for exp 1A ################### 

## contrast coding ##
xp_1A <- filter(all_gen_syllables, xp_num == 1)

#difference fixed effect (different is positive, shared negative)
xp_1A$diffVshared <- ifelse(xp_1A$condition == 'weak-dif' | xp_1A$condition == 'strong-dif',.25,-.25)

#strength fixed effect (strong is positive, weak is negative, shared conditions are 0)
xp_1A$weakVstrong <- ifelse(xp_1A$condition == 'native_shared' | xp_1A$condition == 'NN_shared',0,ifelse(xp_1A$condition=='strong-dif',.5,-.5))

#accent fixed effect (NN shared is positive, native shared is negative, different conditions are 0)
xp_1A$accent <- ifelse(xp_1A$condition == 'NN_shared',.5,ifelse(xp_1A$condition == 'native_shared',-.5,0))


## regression model ##
xp_1a.glm <- glmer(response~legality * (diffVshared + weakVstrong + accent)
                              + (1 + legality|worker_ID) 
                              + (1 + legality|syllable_no_speaker)
                              ,data=xp_1A,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(xp_1a.glm)

## model comparisons ##
chi_sq_legality_main_effect <- chiReport.func(anova(xp_1a.glm,update(xp_1a.glm,.~.-legality)))
chi_sq_shared_main_effect <- chiReport.func(anova(xp_1a.glm,update(xp_1a.glm,.~.-diffVshared)))
chi_sq_accent_main_effect <- chiReport.func(anova(xp_1a.glm,update(xp_1a.glm,.~.-accent)))
chi_sq_strength_main_effect <- chiReport.func(anova(xp_1a.glm,update(xp_1a.glm,.~.-weakVstrong)))
chi_sq_shared_int <- chiReport.func(anova(xp_1a.glm,update(xp_1a.glm,.~.-legality:diffVshared)))
chi_sq_strength_int <- chiReport.func(anova(xp_1a.glm,update(xp_1a.glm,.~.-legality:weakVstrong)))
chi_sq_accent_int <- chiReport.func(anova(xp_1a.glm,update(xp_1a.glm,.~.-legality:accent)))



## write files ##
glm_summary_file_name <- paste0("model_results/per1A_glm.txt")
capture.output(summary(xp_1a.glm)) %>% writeLines(con = glm_summary_file_name)


chi_sq_file_name <- paste0("model_results/per1A_chi_sq.txt")
chi_sq_output <- paste0("legality main effect: ", capture.output(chi_sq_legality_main_effect),
                        "\n shared main effect: ", capture.output(chi_sq_shared_main_effect),
                        "\n accent main effect: ", capture.output(chi_sq_accent_main_effect),
                        "\n strength main effect: ", capture.output(chi_sq_strength_main_effect),
                "\n shared-legality interaction: ", capture.output(chi_sq_shared_int),
                "\n strength-legality interaction: ",capture.output(chi_sq_strength_int),
                "\n accent-legality interaction: ",capture.output(chi_sq_accent_int))
chi_sq_output %>% writeLines(con = chi_sq_file_name)


exp_1A.comp.glm_block <- glmer(response~legality*(diffVshared+accent + block_centered)  + legality:diffVshared:block_centered
                              + (1 + legality|worker_ID) 
                              + (1 + legality|syllable_no_speaker)
                              ,data=xp_1A,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(exp_1A.comp.glm_block)


glm_summary_file_name <- paste0("model_results/per1A_glm_block.txt")
capture.output(summary(exp_1A.comp.glm_block)) %>% writeLines(con = glm_summary_file_name)



################### 4 - model for Exp1B ###################
xp_1B <- filter(all_gen_syllables, xp_num == 2)

## fixed effects ##
#different fixed effect (different is positive coding, shared negative)
xp_1B$diffVshared <- ifelse(xp_1B$condition == 'strong-dif2' | xp_1B$condition == 'weak-dif2',.25,-.5)

#strength fixed effect (strong is positive, weak is negative, shared condition is 0)
xp_1B$weakVstrong <- ifelse(xp_1B$condition == 'NN_shared2',0,ifelse(xp_1B$condition=='strong-dif2',.5,-.5))


## model ##
xp1B_model <- glmer(response~legality*(diffVshared+weakVstrong)
                   + (1 + legality|worker_ID) 
                   + (1 + legality|syllable_no_speaker)
                   ,data=xp_1B,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(xp1B_model)


## model comparisons ##
exp2_chi_sq_legality_main_effect <- capture.output(chiReport.func(anova(xp1B_model,update(xp1B_model,.~.-legality))))
exp2_chi_sq_shared_main_effect <- capture.output(chiReport.func(anova(xp1B_model,update(xp1B_model,.~.-diffVshared))))
exp2_chi_sq_strength_main_effect <- capture.output(chiReport.func(anova(xp1B_model,update(xp1B_model,.~.-weakVstrong))))
exp2_chi_sq_shared_int <- capture.output(chiReport.func(anova(xp1B_model,update(xp1B_model,.~.-legality:diffVshared))))
exp2_chi_sq_strength_int <- capture.output(chiReport.func(anova(xp1B_model,update(xp1B_model,.~.-legality:weakVstrong))))

## write results ##
glm_summary_file_name <- paste0("model_results/per1B_glm_summary.txt")
capture.output(summary(xp1B_model)) %>% writeLines(con = glm_summary_file_name)


chi_sq_file_name <- paste0("model_resultsper1B_chi_sq.txt")
chi_sq_output <- paste0("legality main effect: ", exp2_chi_sq_legality_main_effect, 
                        "\n shared main effect: ", exp2_chi_sq_shared_main_effect,
                        "\n strength main effect: ", exp2_chi_sq_strength_main_effect,
                        "\n shared-legality interaction: ", exp2_chi_sq_shared_int,
                        "\n strength-legality interaction: ",exp2_chi_sq_strength_int)
chi_sq_output %>% writeLines(con = chi_sq_file_name)


exp_1B.comp.glm_block <- glmer(response~legality*(diffVshared + weakVstrong+ block_centered) + legality:diffVshared:block_centered
                               + (1 + legality|worker_ID) 
                               + (1 + legality|syllable_no_speaker)
                               ,data=xp_1B,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(exp_1B.comp.glm_block)

glm_summary_file_name <- paste0("model_results/per1B_glm_block.txt")
capture.output(summary(exp_1B.comp.glm_block)) %>% writeLines(con = glm_summary_file_name)
