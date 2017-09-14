###
# Pre-processing and loading, takes about 10 minutes
###

library(readr)
sig_indicators_cd <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/sig_indicators_L6_cookd.txt", 
                                "\t", escape_double = FALSE, col_names = FALSE)

sig_indicators_df <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/sig_indicators_L6_dffits.txt", 
                                "\t", escape_double = FALSE, col_names = FALSE)

sig_indicators_dsr <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/sig_indicators_L6_dsr.txt", 
                                 "\t", escape_double = FALSE, col_names = FALSE)

sig_indicators_cutie <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/sig_indicators_L6_cutie_1pc.txt", 
                                   "\t", escape_double = FALSE, col_names = FALSE)

initial_SLR_pvalue <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/initial_SLR_ppvalue_L6.txt", 
                                 "\t", escape_double = FALSE, col_names = FALSE)

initial_SLR_pcorr <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/initial_SLR_pcorr_L6.txt", 
                                "\t", escape_double = FALSE, col_names = FALSE)

bact_avg <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/bact_avg_L6.txt", 
                       "\t", escape_double = FALSE, col_names = FALSE)

meta_avg <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/meta_avg_L6.txt", 
                       "\t", escape_double = FALSE, col_names = FALSE)

samp_bact <- read_delim("~/Desktop/Clemente Lab/CUtIe/data/otu_table_MultiO_merged___L6.txt", 
                        "\t", escape_double = FALSE, skip = 1)

samp_meta <- read_delim("~/Desktop/Clemente Lab/CUtIe/data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt", 
                        "\t", escape_double = FALSE)

points_cutie <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/points_L6_['cutie_1pc'].txt", 
                           "\t", escape_double = FALSE, col_names = FALSE)

points_df <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/points_L6_['dffits'].txt", 
                        "\t", escape_double = FALSE, col_names = FALSE)

points_dsr <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/points_L6_['dsr'].txt", 
                         "\t", escape_double = FALSE, col_names = FALSE)

points_cd <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/points_L6_['cookd'].txt", 
                        "\t", escape_double = FALSE, col_names = FALSE)

points_cutie <- points_cutie[,-7]
points_df <- points_df[,-7]
points_dsr <- points_dsr[,-7]
points_cd <- points_cd[,-7]

# drop 110705RC.N.1.RL
samp_meta <- samp_meta[c(1,18:100)]
colnames(samp_meta)[1] <- c("SampleID")
samp_meta <- samp_meta[!(samp_meta$SampleID == "110705RC.N.1.RL"),]
drops <- c("110705RC.N.1.RL")
samp_bact <- samp_bact[ ,!(names(samp_bact) %in% drops)]
samp_bact2 <- t(samp_bact)

target <- names(samp_bact)[-1]
samp_meta = samp_meta[match(target, samp_meta$SampleID),]

colnames(samp_bact)[1] <- c("OTU")
bact_names = samp_bact$OTU
meta_names = colnames(samp_meta)[-1]
# drop the last column
num_meta = ncol(samp_meta) - 1 # 84 - 1
num_bact = nrow(samp_bact) # 897
initial_SLR_pvalue <- initial_SLR_pvalue[,-num_meta-1]
initial_SLR_pcorr <- initial_SLR_pcorr[,-num_meta-1]
bact_avg <- as.numeric(bact_avg[-num_bact-1])
meta_avg <- as.numeric(meta_avg[-num_meta-1])
# get r2 from pcorr
initial_SLR_r2 <- initial_SLR_pcorr * initial_SLR_pcorr
# get log p
logpvals = log(initial_SLR_pvalue)

sig_indicators_cutie <- sig_indicators_cutie[,-num_meta-1]
sig_indicators_cd <- sig_indicators_cd[,-num_meta-1]
sig_indicators_df <- sig_indicators_df[,-num_meta-1]
sig_indicators_dsr <- sig_indicators_dsr[,-num_meta-1]


###
# Duplicated function
###

dupsBetweenGroups <- function (df, idcol) {
  # df: the data frame
  # idcol: the column which identifies the group each row belongs to
  
  # Get the data columns to use for finding matches
  datacols <- setdiff(names(df), idcol)
  
  # Sort by idcol, then datacols. Save order so we can undo the sorting later.
  sortorder <- do.call(order, df)
  df <- df[sortorder,]
  
  # Find duplicates within each id group (first copy not marked)
  dupWithin <- duplicated(df)
  
  # With duplicates within each group filtered out, find duplicates between groups. 
  # Need to scan up and down with duplicated() because first copy is not marked.
  dupBetween = rep(NA, nrow(df))
  dupBetween[!dupWithin] <- duplicated(df[!dupWithin,datacols])
  dupBetween[!dupWithin] <- duplicated(df[!dupWithin,datacols], fromLast=TRUE) | dupBetween[!dupWithin]
  
  # ============= Replace NA's with previous non-NA value ==============
  # This is why we sorted earlier - it was necessary to do this part efficiently
  
  # Get indexes of non-NA's
  goodIdx <- !is.na(dupBetween)
  
  # These are the non-NA values from x only
  # Add a leading NA for later use when we index into this vector
  goodVals <- c(NA, dupBetween[goodIdx])
  
  # Fill the indices of the output vector with the indices pulled from
  # these offsets of goodVals. Add 1 to avoid indexing to zero.
  fillIdx <- cumsum(goodIdx)+1
  
  # The original vector, now with gaps filled
  dupBetween <- goodVals[fillIdx]
  
  # Undo the original sort
  dupBetween[sortorder] <- dupBetween
  
  # Return the vector of which entries are duplicated across groups
  return(dupBetween)
}

###
# Find desired overlap
###

points_cutie$metric <- "CUtIe"
points_cd$metric <- "CookD"
points_df$metric <- "Dffits"
points_dsr$metric <- "Dsr"

df <- rbind(points_cd, points_cutie, points_df, points_dsr)
df <- df[,c("metric", "X1", "X2","X3", "X4", "X5", "X6")] 
dupRows <- dupsBetweenGroups(df[,1:5], "metric")
df <- cbind(df, dup=dupRows)
union_points <- df[df$dup == TRUE,]
union_points <- union_points[,1:7]

cutie_only <- rbind(points_cutie, union_points)
cutie_only <- cutie_only[,c("metric", "X1", "X2", "X3", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(cutie_only, "metric")
cutie_only <- cbind(cutie_only, dup=dupRows)
cutie_only <- cutie_only[cutie_only$dup == FALSE,]

df_only <- rbind(points_df, union_points)
df_only <- df_only[,c("metric", "X1", "X2", "X3", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(df_only, "metric")
df_only <- cbind(df_only, dup=dupRows)
df_only <- df_only[df_only$dup == FALSE,]

dsr_only <- rbind(points_dsr, union_points)
dsr_only <- dsr_only[,c("metric", "X1", "X2", "X3", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(dsr_only, "metric")
dsr_only <- cbind(dsr_only, dup=dupRows)
dsr_only <- dsr_only[dsr_only$dup == FALSE,]

cd_only <- rbind(points_cd, union_points)
cd_only <- cd_only[,c("metric", "X1", "X2", "X3", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(cd_only, "metric")
cd_only <- cbind(cd_only, dup=dupRows)
cd_only <- cd_only[cd_only$dup == FALSE,]

c(nrow(cutie_only),nrow(cd_only),nrow(dsr_only),nrow(df_only))

###
# Correlations only
### 

# drop samples
corr_cutie = points_cutie[,c("X1","X2","X4","X5","X6","metric")]
corr_cd = points_cd[,c("X1","X2","X4","X5","X6","metric")]
corr_df = points_df[,c("X1","X2","X4","X5","X6","metric")]
corr_dsr = points_dsr[,c("X1","X2","X4","X5","X6","metric")]

corr_cutie = corr_cutie[!duplicated(corr_cutie[,c("X1","X2")]), ]
corr_cd = corr_cd[!duplicated(corr_cd[,c("X1","X2")]), ]
corr_df = corr_df[!duplicated(corr_df[,c("X1","X2")]), ]
corr_dsr = corr_dsr[!duplicated(corr_dsr[,c("X1","X2")]), ]

df <- rbind(corr_cd, corr_cutie, corr_df, corr_dsr)
df <- df[,c("metric", "X1", "X2", "X4", "X5", "X6")] 
dupRows <- dupsBetweenGroups(df[,1:3], "metric")
df <- cbind(df, dup=dupRows)
union_corr <- df[df$dup == TRUE,]
union_corr <- union_corr[,1:6]

cutiecorr_only <- rbind(corr_cutie, union_corr)
cutiecorr_only <- cutiecorr_only[,c("metric", "X1", "X2", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(cutiecorr_only[,1:3], "metric")
cutiecorr_only <- cbind(cutiecorr_only, dup=dupRows)
cutiecorr_only <- cutiecorr_only[cutiecorr_only$dup == FALSE,]

dfcorr_only <- rbind(corr_df, union_corr)
dfcorr_only <- dfcorr_only[,c("metric", "X1", "X2", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(dfcorr_only[,1:3], "metric")
dfcorr_only <- cbind(dfcorr_only, dup=dupRows)
dfcorr_only <- dfcorr_only[dfcorr_only$dup == FALSE,]

dsrcorr_only <- rbind(corr_dsr, union_corr)
dsrcorr_only <- dsrcorr_only[,c("metric", "X1", "X2", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(dsrcorr_only[,1:3], "metric")
dsrcorr_only <- cbind(dsrcorr_only, dup=dupRows)
dsrcorr_only <- dsrcorr_only[dsrcorr_only$dup == FALSE,]

cdcorr_only <- rbind(corr_cd, union_corr)
cdcorr_only <- cdcorr_only[,c("metric", "X1", "X2", "X4", "X5", "X6")]
dupRows <- dupsBetweenGroups(cdcorr_only[,1:3], "metric")
cdcorr_only <- cbind(cdcorr_only, dup=dupRows)
cdcorr_only <- cdcorr_only[cdcorr_only$dup == FALSE,]

c(nrow(cutiecorr_only),nrow(cdcorr_only),nrow(dsrcorr_only),nrow(dfcorr_only))

###
# All but COOKD, All but CUtIe
###

setwd("/Users/kbpi31415/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05")

pairs_L6_cutie_1pc_dffits_dsr_ <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/pairs_L6_['cutie_1pc', 'dffits', 'dsr'].txt", 
                                              "\t", escape_double = FALSE, col_names = FALSE)
pairs_L6_cookd_dffits_dsr_ <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/pairs_L6_['cookd', 'dffits', 'dsr'].txt", 
                                          "\t", escape_double = FALSE, col_names = FALSE)

pairs_L6_cutie_1pc_cookd_dffits_dsr_ <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/pairs_L6_['cutie_1pc', 'cookd', 'dffits', 'dsr'].txt", 
                                                    "\t", escape_double = FALSE, col_names = FALSE)

pairs_L6_cutie_1pc_dffits_dsr_ = pairs_L6_cutie_1pc_dffits_dsr_[,-3] 
pairs_L6_cookd_dffits_dsr_ = pairs_L6_cookd_dffits_dsr_[,-3] 
pairs_L6_cutie_1pc_cookd_dffits_dsr_ = pairs_L6_cutie_1pc_cookd_dffits_dsr_[,-3] 

pairs_L6_cutie_1pc_dffits_dsr_$metric = "noncookd"
pairs_L6_cookd_dffits_dsr_$metric = "noncutie"
pairs_L6_cutie_1pc_cookd_dffits_dsr_$metric = "all"

points_L6_cutie_1pc_dffits_dsr_ <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/points_L6_['cutie_1pc', 'dffits', 'dsr'].txt", 
                                             "\t", escape_double = FALSE, col_names = FALSE)
points_L6_cookd_dffits_dsr_ <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/points_L6_['cookd', 'dffits', 'dsr'].txt", 
                                         "\t", escape_double = FALSE, col_names = FALSE)

points_L6_cutie_1pc_cookd_dffits_dsr_ <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_pointcomparison_pc1fdr0.05/data_processing/points_L6_['cutie_1pc', 'cookd', 'dffits', 'dsr'].txt", 
                                                   "\t", escape_double = FALSE, col_names = FALSE)

points_L6_cutie_1pc_dffits_dsr_ = points_L6_cutie_1pc_dffits_dsr_[,-7] 
points_L6_cookd_dffits_dsr_ = points_L6_cookd_dffits_dsr_[,-7] 
points_L6_cutie_1pc_cookd_dffits_dsr_ = points_L6_cutie_1pc_cookd_dffits_dsr_[,-7] 

points_L6_cutie_1pc_dffits_dsr_$metric = "noncookd"
points_L6_cookd_dffits_dsr_$metric = "noncutie"
points_L6_cutie_1pc_cookd_dffits_dsr_$metric = "all"



# pall but cookd
pallbutcookd <- rbind(points_L6_cutie_1pc_dffits_dsr_, points_L6_cutie_1pc_cookd_dffits_dsr_)
pallbutcookd <- pallbutcookd[,c("metric", "X1", "X2","X3")]
dupRows <- dupsBetweenGroups(pallbutcookd, "metric")
pallbutcookd <- cbind(pallbutcookd, dup=dupRows)
pallbutcookd <- pallbutcookd[pallbutcookd$dup == FALSE,]


# pall but cutie
pallbutcutie <- rbind(points_L6_cookd_dffits_dsr_, points_L6_cutie_1pc_cookd_dffits_dsr_)
pallbutcutie <- pallbutcutie[,c("metric", "X1", "X2","X3")]
dupRows <- dupsBetweenGroups(pallbutcutie, "metric")
pallbutcutie <- cbind(pallbutcutie, dup=dupRows)
pallbutcutie <- pallbutcutie[pallbutcutie$dup == FALSE,]



# all but cutie
allbutcutie <- rbind(pairs_L6_cookd_dffits_dsr_, pairs_L6_cutie_1pc_cookd_dffits_dsr_)
allbutcutie <- allbutcutie[,c("metric", "X1", "X2")]
dupRows <- dupsBetweenGroups(allbutcutie, "metric")
allbutcutie <- cbind(allbutcutie, dup=dupRows)
allbutcutie <- allbutcutie[allbutcutie$dup == FALSE,]



# all but cookd
allbutcookd <- rbind(points_L6_cutie_1pc_dffits_dsr_, points_L6_cutie_1pc_cookd_dffits_dsr_)
allbutcookd <- allbutcookd[,c("metric", "X1", "X2")]
dupRows <- dupsBetweenGroups(allbutcookd, "metric")
allbutcookd <- cbind(allbutcookd, dup=dupRows)
allbutcookd <- allbutcookd[allbutcookd$dup == FALSE,]

###
# SPECIFIC ANALYSIS HERE
###


c(nrow(cutie_only[cutie_only$X4 < 0.0005,]), 
  nrow(cd_only[cd_only$X4 < 0.0005,]), 
  nrow(df_only[df_only$X4 < 0.0005,]), 
  nrow(dsr_only[dsr_only$X4 < 0.0005,]))

c(nrow(cutie_only[cutie_only$X4 > 0.05,]), 
  nrow(cd_only[cd_only$X4 > 0.05,]), 
  nrow(df_only[df_only$X4 > 0.05,]), 
  nrow(dsr_only[dsr_only$X4 > 0.05,]))


# distribution of p-values
hist(log(cutie_only$X6))
hist(log(cd_only$X6))
hist(log(df_only$X6))
hist(log(dsr_only$X6))



# for triplet
point1 = 1
point2 = 13
point3 = 15
plot(as.numeric(as.vector(t(samp_bact)[,point1][-1])),as.numeric(as.vector(samp_meta[,point2+1])), 
     xlab=paste("bact", point1, " abundance"), ylab=paste("meta", point2, " abundance"),
     main=paste("r2 = ", floor(initial_SLR_r2[point1,point2]*100)/100, " logp = ",
                floor(log(initial_SLR_pvalue[point1,point2])*100)/100))
abline(lm(as.numeric(as.vector(samp_meta[,point2+1])) ~ as.numeric(as.vector(t(samp_bact)[,point1][-1]))))

points_cutie[(points_cutie$X1 == point1-1 & points_cutie$X2 == point2-1),]
model <- lm(as.numeric(as.vector(samp_meta[,point2+1])) ~ as.numeric(as.vector(t(samp_bact)[,point1][-1])))

cdx = cooks.distance(model)
cdx[point3]
dsrx = rstudent(model)
dsrx[point3]
dfx = dffits(model)
dfx[point3]


dir.create('graphs')
dir.create('graphs/allbutcutie')
for (i in 1:min(100,nrow(allbutcutie)))
{
  bact = allbutcutie$X1[i] + 1
  meta = allbutcutie$X2[i] + 1
  fit1 = lm(as.numeric(as.vector(samp_meta[,meta+1])) ~ as.numeric(as.vector(t(samp_bact)[,bact][-1])))
  r1 = summary(fit1)$r.squared
  p1 = summary(fit1)$coefficients[,4][2]
  pdf(paste('graphs/allbutcutie/allbutcutie_',bact_names[bact],'_',meta_names[meta],'.pdf'), width=3, height=3.5)
  plot(as.numeric(as.vector(t(samp_bact)[,bact][-1])),
       as.numeric(as.vector(samp_meta[,meta+1])), 
       xlab = paste(bact,' abundance'),
       ylab = paste(meta, ' level'), 
       main=paste("r2 = ", floor(r1*1e5)/1e5, " log p = ",floor(log(p1)*1e5)/1e5))
  abline(fit1)
  dev.off()
}

dir.create('graphs')
dir.create('graphs/allbutcookd')
for (i in 1:min(100,nrow(allbutcookd)))
{
  bact = allbutcookd$X1[i] + 1
  meta = allbutcookd$X2[i] + 1
  fit1 = lm(as.numeric(as.vector(samp_meta[,meta+1])) ~ as.numeric(as.vector(t(samp_bact)[,bact][-1])))
  r1 = summary(fit1)$r.squared
  p1 = summary(fit1)$coefficients[,4][2]
  pdf(paste('graphs/allbutcookd/allbutcookd',bact_names[bact],'_',meta_names[meta],'.pdf'), width=3, height=3.5)
  plot(as.numeric(as.vector(t(samp_bact)[,bact][-1])),
       as.numeric(as.vector(samp_meta[,meta+1])), 
       xlab = paste(bact,' abundance'),
       ylab = paste(meta, ' level'), 
       main=paste("r2 = ", floor(r1*1e5)/1e5, " log p = ",floor(log(p1)*1e5)/1e5))
  abline(fit1)
  dev.off()
}



dir.create('graphs/cutieonly')
for (i in 1:min(100,nrow(cutie_only)))
{
  bact = cutie_only$X1[i] + 1
  meta = cutie_only$X2[i] + 1
  fit1 = lm(as.numeric(as.vector(samp_meta[,meta+1])) ~ as.numeric(as.vector(t(samp_bact)[,bact][-1])))
  r1 = summary(fit1)$r.squared
  p1 = summary(fit1)$coefficients[,4][2]
  pdf(paste('graphs/cutieonly/cutieonly_',bact_names[bact],'_',meta_names[meta],'.pdf'), width=3, height=3.5)
  plot(as.numeric(as.vector(t(samp_bact)[,bact][-1])),
       as.numeric(as.vector(samp_meta[,meta+1])), 
       xlab = paste('bact', bact, ' abundance'),
       ylab = paste('meta ', meta, ' level'), 
       main=paste("r2 = ", floor(r1*1e5)/1e5, " log p = ",floor(log(p1)*1e5)/1e5))
  abline(fit1)
  dev.off()
}


