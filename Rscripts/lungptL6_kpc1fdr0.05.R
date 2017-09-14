###
# Pre-processing and loading, takes about 10 minutes
###

library(readr)
sig_indicators_k1 <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc3fdr0.05/data_processing/sig_indicators_L6_resample1.txt", 
                                "\t", escape_double = FALSE, col_names = FALSE)

sig_indicators_k2 <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc3fdr0.05/data_processing/sig_indicators_L6_resample2.txt", 
                                "\t", escape_double = FALSE, col_names = FALSE)

sig_indicators_k3 <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc3fdr0.05/data_processing/sig_indicators_L6_resample3.txt", 
                                "\t", escape_double = FALSE, col_names = FALSE)

initial_SLR_pvalue <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc3fdr0.05/data_processing/initial_SLR_ppvalue_L6.txt", 
                                 "\t", escape_double = FALSE, col_names = FALSE)

initial_SLR_pcorr <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc3fdr0.05/data_processing/initial_SLR_pcorr_L6.txt", 
                                "\t", escape_double = FALSE, col_names = FALSE)

bact_avg <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc3fdr0.05/data_processing/bact_avg_L6.txt", 
                       "\t", escape_double = FALSE, col_names = FALSE)

meta_avg <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc3fdr0.05/data_processing/meta_avg_L6.txt", 
                       "\t", escape_double = FALSE, col_names = FALSE)

samp_bact <- read_delim("~/Desktop/Clemente Lab/CUtIe/data/otu_table_MultiO_merged___L6.txt", 
                        "\t", escape_double = FALSE, skip = 1)

samp_meta <- read_delim("~/Desktop/Clemente Lab/CUtIe/data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt", 
                        "\t", escape_double = FALSE)

R_matrix_L6 <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc1fdr0.05/data_processing/R_matrix_L6.txt", 
                          "\t", escape_double = FALSE)

raw_data <- R_matrix_L6[,-9]

# drop 110705RC.N.1.RL
samp_meta <- samp_meta[c(1,18:100)]
colnames(samp_meta)[1] <- c("SampleID")
samp_meta <- samp_meta[!(samp_meta$SampleID == "110705RC.N.1.RL"),]
drops <- c("110705RC.N.1.RL")
samp_bact <- samp_bact[ ,!(names(samp_bact) %in% drops)]

colnames(samp_bact)[1] <- c("OTU")
bact_names = samp_bact$OTU
meta_names = colnames(samp_meta)[-1]
samp_names = colnames(samp_bact)[-1]

samp_meta = samp_meta[match(samp_names, samp_meta$SampleID),]

# drop the last column
num_meta = ncol(samp_meta) - 1 # 84 - 1
num_bact = nrow(samp_bact) # 897
initial_SLR_pvalue <- initial_SLR_pvalue[,-num_meta-1]
initial_SLR_pcorr <- initial_SLR_pcorr[,-num_meta-1]
bact_avg <- as.numeric(bact_avg[-num_bact-1])
meta_avg <- as.numeric(meta_avg[-num_meta-1])
# get r2 from pcorr
initial_SLR_r2 <- initial_SLR_pcorr * initial_SLR_pcorr

### 
# START HERE FOR SPECIFIC ANALYSES; run everything above before starting
###

pairs = raw_data[raw_data$r2vals != 0, ]

TN = pairs[pairs$indicators == 0,]
FP = pairs[pairs$indicators == -1,]
TP = pairs[pairs$indicators == 1,]
P = pairs[pairs$indicators != 0,]

### 
# Plot Generation
###
makeTransparent<-function(someColor, alpha=100)
{
  newColor<-col2rgb(someColor)
  apply(newColor, 2, function(curcoldata){rgb(red=curcoldata[1], green=curcoldata[2],
                                              blue=curcoldata[3],alpha=alpha, maxColorValue=255)})
}


# TP + FP scatterplot with TP as red
symbols(x=P$r2vals, y=P$logp, xlab = "R2", ylab = "logp", main="TP1 (65) and FP1 (2681), bact",
        circles=sqrt(P$avg_bact/pi), fg = ifelse(P$indicators == 1,'red',makeTransparent('black',30)),inches=FALSE)
c(min(P$avg_bact), max(P$avg_bact))

symbols(x=P$r2vals, y=P$logp, xlab = "R2", ylab = "logp", main="TP and FP, meta",
        circles=sqrt(P$avg_meta/pi), fg = ifelse(P$indicators == 1,'red',makeTransparent('black',30)),inches=1/3)
c(min(P$avg_meta), max(P$avg_meta))


# Genus analysis, most common species
sort(table(FP$bact_index),decreasing=TRUE)[1:10]
length(unique(FP$bact_index)) 


# for pair (457, 13)
point1 = 458
point2 = 26
plot(as.numeric(as.vector(t(samp_bact)[,point1][-1])),as.numeric(as.vector(samp_meta[,point2+1])), 
     xlab=paste("bact", point1, " abundance"), ylab=paste("meta", point2, " abundance"),
     main=paste("r2 = ", floor(initial_SLR_r2[point1,point2]*100)/100, " logp = ",
                floor(log(initial_SLR_pvalue[point1,point2])*100)/100))
abline(lm(as.numeric(as.vector(samp_meta[,point2+1])) ~ as.numeric(as.vector(t(samp_bact)[,point1][-1]))))


point1 = 460
point2 = 17
plot(as.numeric(as.vector(t(samp_bact)[,point1][-1])),as.numeric(as.vector(t(samp_bact)[,point2][-1])), 
     xlab=paste("bact", point1, " abundance"), ylab=paste("bact", point2, " abundance"))
abline(lm(as.numeric(as.vector(t(samp_bact)[,point2][-1])) ~ as.numeric(as.vector(t(samp_bact)[,point1][-1]))))


###
# Mass graph generation
###

setwd("/Users/kbpi31415/Desktop/Clemente Lab/CUtIe/data_analysis/lungptL6_kpc1fdr0.05")
dir.create('graphs')
dir.create('graphs/TP')

for (i in 1:min(100,nrow(TP)))
{
  bact = TP$bact_index[i]
  meta = TP$meta_index[i]
  fit1 = lm(as.numeric(as.vector(samp_meta[,meta+2])) ~ as.numeric(as.vector(t(samp_bact)[,bact+1][-1])))
  r1 = summary(fit1)$r.squared
  p1 = summary(fit1)$coefficients[,4][2]
  pdf(paste('graphs/TP/TP_',bact_names[bact+1],'_',meta_names[meta+1],'.pdf'), width=3, height=3.5)
  plot(as.numeric(as.vector(t(samp_bact)[,bact+1][-1])),
       as.numeric(as.vector(samp_meta[,meta+2])), 
       xlab = paste(bact+1,' abundance'),
       ylab = paste(meta+1, ' level'), 
       main=paste("r2 = ", floor(r1*1e5)/1e5, " log p = ",floor(log(p1)*1e5)/1e5))
  abline(fit1)
  dev.off()
}

dir.create('graphs/FP')
for (i in 1:min(100,nrow(FP)))
{
  bact = FP$bact_index[i]
  meta = FP$meta_index[i]
  fit1 = lm(as.numeric(as.vector(samp_meta[,meta+2])) ~ as.numeric(as.vector(t(samp_bact)[,bact+1][-1])))
  r1 = summary(fit1)$r.squared
  p1 = summary(fit1)$coefficients[,4][2]
  pdf(paste('graphs/FP/FP_',bact_names[bact+1],'_',meta_names[meta+1],'.pdf'), width=3, height=3.5)
  plot(as.numeric(as.vector(t(samp_bact)[,bact+1][-1])),
       as.numeric(as.vector(samp_meta[,meta+2])), 
       xlab = paste(bact+1,' abundance'),
       ylab = paste(meta+1, ' level'), 
       main=paste("r2 = ", floor(r1*1e5)/1e5, " log p = ",floor(log(p1)*1e5)/1e5))
  abline(fit1)
  dev.off()
}
