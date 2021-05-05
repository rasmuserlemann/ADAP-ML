rm(list=ls())
library(mixOmics)

data = read.csv('SCLC_study_output_filtered_2.csv')
data = subset(data, select=-1)
resp = read.csv('SCLC_study_responses_2.csv')
resp = resp$resp

pls = plsda(data, resp)