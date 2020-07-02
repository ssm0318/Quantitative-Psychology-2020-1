
update.packages(ask = FALSE)

## libraries
install.packages("ggcorrplot", dependencies = TRUE)
install.packages("corrplot", dependencies = TRUE)

install.packages("ggplot2")

library(ggcorrplot)
library(corrplot)

## read data
raw_data <- read.csv("raw_data.csv")
head(raw_data)

agg_data <- read.csv("results_compilation.csv")
head(agg_data)

labels <- read.csv("scale_labels.csv", header = TRUE)

complete_data <- agg_data[agg_data$Complete == "TRUE",]

self_eval <- complete_data 

attach(complete_data)

complete_data[20:64] <- lapply(complete_data[20:64], as.numeric)
scales <- complete_data[, 20:64]


##########################
##########################

## correlation

corr <- cor(scales, use = "complete.obs")
head(corr[, 1:5])

# matrix of p-values 
p.mat <- cor_pmat(scales)
p.mat <- cor.mtest(scales) # same as the above
head(p.mat[, 1:4])

# correlation matrix 1
ggcorrplot(corr)

# correlation matrix 2
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

# correlation matrix 3
corrplot(corr, type="upper", order="hclust", p.mat = p.mat, sig.level = 0.05, insig = "blank", diag = FALSE)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(corr, method="color", col=col(200),  
         type="upper", order="hclust", 
         # Combine with significance
         p.mat = p.mat, sig.level = 0.05, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)


##########################
##########################

## correlation pairs

# boredom proneness ~ ADHD : 0.346 (.00681)
corr1 <- cor.test(ADHD_RS, BPS_SR)
corr1

# external BP ~ ADHD : 0.487 (7.92e-05)
corr2 <- cor.test(ADHD_RS, BPSSR_E)
corr2

# internal BP ~ ADHD : 0.180 (0.168); NOT significant
# same for BPSSR_I, BPSSR_Amb
corr3 <- cor.test(ADHD_RS, BPSSR_I)
corr3

# ADHD ~ risk propensity : 0.415 (0.000877)
corr4 <- cor.test(ADHD_RS, RPS)
corr4

# ADHD ~ boredom susceptibility : 0.377 (0.00300)
corr5 <- cor.test(ADHD_RS, ZBS)
corr5

# ADHD ~ procrastination : 0.590 (5.53e-07)
corr6 <- cor.test(ADHD_RS, PCS_9)
corr6

# ADHD ~ emotional regulation : -0.289 (0.0239)
# subscales were NOT significant
corr7 <- cor.test(ADHD_RS, ERQ_10)
corr7

# ADHD ~ flourishing : -0.361 (0.00429)
corr8 <- cor.test(ADHD_RS, FS)
corr8

# ADHD ~ aggression : 0.497 (4.663e-05) 
corr9 <- cor.test(ADHD_RS, BPAQ_SF)
corr9

# ADHD ~ prevention focus : -0.557 (3.119e-06)
corr10 <- cor.test(ADHD_RS, RFQ_v)
corr10

# ADHD ~ promotion focus : 0.329 (0.00964)
corr11 <- cor.test(ADHD_RS, RFQ_m)
corr11

# ADHD ~ locus of control : 0.260 (0.0446)
# higher LOC_13 score implies external locus of control
corr12 <- cor.test(ADHD_RS, LOC_13)
corr12

# ADHD ~ cognitive/affective mindfulness : -0.576 (1.487e-06)
corr13 <- cor.test(ADHD_RS, CAMS_R)
corr13

# ADHD ~ boredom susceptibility : 0.377 (0.00299)
corr14 <- cor.test(ADHD_RS, ZBS)
corr14

# ADHD ~ internet addiction : 0.496 (4.86e-05)
corr15 <- cor.test(ADHD_RS, SAS)
corr15

corr <- cor.test(ADHD_RS, SOAS)
corr

corr_E <- cor.test(BPSSR_E, ERQ_10_e)
corr_I <- cor.test(BPSSR_I, ERQ_10_e)
corr_E
corr_I



