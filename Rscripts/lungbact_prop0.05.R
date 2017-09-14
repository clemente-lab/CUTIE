library(readr)
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

prop <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungbact_prop0.05/data_processing/prop.txt", 
                   "\t", escape_double = FALSE)

sig_indicators_cutie_propL6_resample1 <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungbact_prop0.05/data_processing/sig_indicators_cutie_propL6_resample1.txt", 
                                                    "\t", escape_double = FALSE, col_names = FALSE, skip = 1)

samp_bact_mr <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungbact_prop0.05/data_processing/samp_bact_mr.txt", 
                           "\t", escape_double = FALSE)

R_matrix_L6 <- read_delim("~/Desktop/Clemente Lab/CUtIe/data_analysis/lungbact_prop0.05/data_processing/R_matrix_L6.txt", 
                          "\t", escape_double = FALSE)

R_matrix_L6 <- R_matrix_L6[,-7]
samp_bact_mr <- samp_bact_mr[,-898]

P = R_matrix_L6[R_matrix_L6$indicators != 0,]
TP = R_matrix_L6[R_matrix_L6$indicators == 1,]
FP = R_matrix_L6[R_matrix_L6$indicators == -1,]


point1 = 353
point2 = 746
plot(as.numeric(as.vector(samp_bact_mr[,point1])),as.numeric(as.vector(samp_bact_mr[,point2])), 
     xlab=paste("bact1", point1, " abundance"), ylab=paste("bact2", point2, " abundance"),
     main=paste("prop = ", floor(prop[point1,point2]*100)/100))
abline(lm(as.numeric(as.vector(samp_bact_mr[,point2])) ~ as.numeric(as.vector(samp_bact_mr[,point1]))))



setwd("/Users/kbpi31415/Desktop/Clemente Lab/CUtIe/data_analysis/lungbact_prop0.05")
dir.create('graphs')
dir.create('graphs/TP')

for (i in 1:min(100,nrow(TP)))
{
  bact1 = TP$bact1_index[i] + 1
  bact2 = TP$bact2_index[i] + 1
  fit1 = lm(as.numeric(as.vector(samp_bact_mr[,bact2])) ~ as.numeric(as.vector(samp_bact_mr[,bact1])))
  r1 = summary(fit1)$r.squared
  p1 = summary(fit1)$coefficients[,4][2]
  pdf(paste('graphs/TP/TP_',bact_names[bact1+1],'_',bact_names[bact2+1],'.pdf'), width=3, height=3.5)
  plot(as.numeric(as.vector(samp_bact_mr[,bact1])),as.numeric(as.vector(samp_bact_mr[,bact2])), 
       xlab=paste("bact1", bact1, " abundance"), ylab=paste("bact2", bact2, " abundance"),
       main=paste("prop = ", floor(prop[bact1,bact2]*100)/100))
  abline(fit1)
  dev.off()
}

dir.create('graphs/FP')
for (i in 1:min(100,nrow(FP)))
{
  bact1 = FP$bact1_index[i] + 1 
  bact2 = FP$bact2_index[i] + 1
  fit1 = lm(as.numeric(as.vector(samp_bact_mr[,bact2])) ~ as.numeric(as.vector(samp_bact_mr[,bact1])))
  r1 = summary(fit1)$r.squared
  p1 = summary(fit1)$coefficients[,4][2]
  pdf(paste('graphs/FP/FP_',bact_names[bact1+1],'_',bact_names[bact2+1],'.pdf'), width=3, height=3.5)
  plot(as.numeric(as.vector(samp_bact_mr[,bact1])),as.numeric(as.vector(samp_bact_mr[,bact2])), 
       xlab=paste("bact1", bact1, " abundance"), ylab=paste("bact2", bact2, " abundance"),
       main=paste("prop = ", floor(prop[bact1,bact2]*100)/100))
  abline(fit1)
  dev.off()
}


