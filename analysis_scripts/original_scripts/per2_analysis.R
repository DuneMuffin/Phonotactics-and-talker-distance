############################################################
# Thomas Denby - 2/22/2019
# Rational Inferece in Phonotactic Learning
# osf.io/a6pjv/
# 
# This is the analysis script for Experiment 2
# 
# Sections include: 
#                  (1) data prep
#                  (2) means (familiarization, legality, legality by speaker, vowel, etc.)
#                  (3) Exp2 model
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
#       if you're sourcing the script, uncomment the appropriate section below (line 38).
#       alternatively, you can manually enter in the script's filepath on your computer
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
# dirname(sys.frame(1)$ofile)

#manually (fill in the path to the folder including this script)
# setwd('local_path_here/results_scripts')

#load functions from script
source("analysis_functions.R")

################### 1 - data prep ################### 

#open file
per2_file_path <- 'data/per2_filteredSubjects.csv'
per2_data <- open_file(per2_file_path, experiment = 2)

#split into generalization (or gen) syllables vs. familiarization (fam) syllables
all_gen_syllables <- filter(per2_data, gen == 0.5)
all_fam_syllables <- filter(per2_data, gen == -0.5)

#turn response, workerID, syllable etc. into factors
all_gen_syllables$response <- as.factor(all_gen_syllables$response)
all_gen_syllables$worker_ID <- as.factor(all_gen_syllables$worker_ID)
all_gen_syllables$syllable <- as.factor(all_gen_syllables$syllable)
all_gen_syllables$syllable_no_speaker <- as.factor(all_gen_syllables$syllable_no_speaker)

#add column of condition names matching condition numbers, add column for condition_num
#experimental list numbers and associated experiments

condition_names <- data.frame(c(1,2,3), c("mixed-different","non-native-different","non-native-shared"))
colnames(condition_names) <- c("condition_num","condition")
all_gen_syllables <- merge(all_gen_syllables,condition_names)

#add column for speaker, gender, vowel, language background, and accent (French lang background vs. English)
all_gen_syllables$label <- as.character(all_gen_syllables$label)
all_gen_syllables$speaker <- substring(all_gen_syllables$label,nchar(all_gen_syllables$label)-3,nchar(all_gen_syllables$label))
all_gen_syllables$gender <- substring(all_gen_syllables$label,nchar(all_gen_syllables$label),nchar(all_gen_syllables$label))
all_gen_syllables$gender_contrast <- ifelse(all_gen_syllables$gender == 'F',0.5,-0.5)
all_gen_syllables$accent <- substring(all_gen_syllables$label,nchar(as.character(all_gen_syllables$label))-3,nchar(as.character(all_gen_syllables$label))-2)



################### 2 - means ################### 
###false alarm rate on familiarization syllables###
fam_means <- fam_mean_CI(all_fam_syllables)

### overall mean false alarm rate and mean legality advantage (FA on legal minus FA on illegal) 
# across participants (not including condition)###
means.overall_FA <- data.frame()
means.legality <- data.frame()

overall_FA_bySub <-  ddply(all_gen_syllables,.(worker_ID),summarize,PropYes = mean(yesResponse))
means.overall_FA <- ddply(overall_FA_bySub,.(), summarize,meanPropYes = mean(PropYes))

#grab confidence intervals for overall false alarms
means.overall_FA <- get_boot_CIs(overall_FA_bySub, means.overall_FA)

# mean legality advantage  #
differencesBySub <- ddply(all_gen_syllables,.(worker_ID,condition),summarize,PropYes = mean(yesResponse[legality == 0.5]) - mean(yesResponse[legality == -0.5]))
mean.differencesBySub <- ddply(differencesBySub,.(),summarize,PropYes = mean(PropYes))
mean.differencesBySub <- get_boot_CIs(differencesBySub, mean.differencesBySub)

means.overall_FA


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
#block/condition
differencesBlockCondition <- ddply(all_gen_syllables,.(condition_num,block_centered),summarize,PropYes = mean(yesResponse)) %>% na.omit()

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

#means by accent
accent_all_means <- ddply(all_gen_syllables,.(condition_num,accent,legality),summarize,PropYes = mean(yesResponse)) %>% na.omit()

#write means
write.table(fam_means, file = "means/per2_experiment_fam_means.csv",sep = ",",row.names = FALSE)
write.table(means.overall_FA, file = "means/per2_experiment_overall_means.csv",sep = ",",row.names = FALSE)
write.table(means.genBySub, file = "means/per2_condition_means.csv",sep = ",",row.names = FALSE)
write.table(diffs.genBySub, file = "means/per2_subj_legality.csv",sep = ",",row.names = FALSE)
write.table(mean_diffs.genBySub, file = "means/per2_condition_legality_means.csv",sep = ",",row.names = FALSE)
write.table(speaker_diff, file = "means/per2_speaker_legality_means.csv",sep = ",",row.names = FALSE)
write.table(speaker_diff_condition, file = "means/per2_speaker_condition_legality_means.csv",sep = ",",row.names = FALSE)




################### 3 - model ################### 
## contrast coding ##

#different is positive coding, shared negative
all_gen_syllables$diffVshared <- ifelse(all_gen_syllables$condition == 'mixed-different' | all_gen_syllables$condition == 'non-native-different',.25,-.5)

#native is positive, non-native is negative
all_gen_syllables$native <- ifelse(all_gen_syllables$condition  == 'mixed-different', .5, -.25)

## model ##
exp2.glm <- glmer(response~legality*(diffVshared+native) + 
                          (1 + legality|worker_ID)+ (1 + legality|syllable_no_speaker)
                          ,data=all_gen_syllables,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(exp2.glm)

chi_sq_legality_main_effect <- chiReport.func(anova(exp2.glm,update(exp2.glm,.~.-legality)))
chi_sq_diffVshared_main_effect <- chiReport.func(anova(exp2.glm,update(exp2.glm,.~.-diffVshared)))
chi_sq_native_main_effect <- chiReport.func(anova(exp2.glm,update(exp2.glm,.~.-native)))
chi_sq_shared_int <- chiReport.func(anova(exp2.glm,update(exp2.glm,.~.-legality:diffVshared)))
chi_sq_native_int <- chiReport.func(anova(exp2.glm,update(exp2.glm,.~.-legality:native)))



# write files
glm_summary_file_name <- paste0("model_results/per2_glm_summary.txt")
capture.output(summary(exp2.glm)) %>% writeLines(con = glm_summary_file_name)


chi_sq_file_name <- paste0("model_results/per2_chi_sq.txt")
chi_sq_output <- paste0("legality main effect: ", capture.output(chi_sq_legality_main_effect),
                        "\n diffVshared main effect: ", capture.output(chi_sq_diffVshared_main_effect),
                        "\nnative main effect: ", capture.output(chi_sq_native_main_effect),
                "\n shared-legality interaction: ", capture.output(chi_sq_shared_int),
                "\n native-legality interaction: ",capture.output(chi_sq_native_int))
chi_sq_output %>% writeLines(con = chi_sq_file_name)



## model with block ##
exp2.glm.block <- glmer(response~legality*(diffVshared+native + block_centered) + legality:diffVshared:block_centered
                              + (1 + legality|worker_ID) 
                              + (1 + legality|syllable_no_speaker)
                              ,data=all_gen_syllables,family="binomial",control=glmerControl(optimizer="bobyqa"))

summary(exp2.glm.block)


glm_block_summary_file_name <- paste0("model_results/per2_glm_summary_block.txt")
capture.output(summary(exp2.glm.block)) %>% writeLines(con = glm_block_summary_file_name)
