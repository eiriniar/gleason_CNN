
rm(list=ls())
library(survival)
library(OIsurv)
library(foreign)
library(survminer)

input_directory <- '/data3/eiriniar/gleason_CNN/dataset_TMA/results'
choices <- c('st_os_gen', 'st_os_spec', 'rfs_status')
names <- c('Overall survival', 'Disease-specific survival', 'Recurrence-free survival')


draw_survival_curves <- function(KM_fit, pvals, annotator, surv_type, surv_name) {
  # make the plot
  ggsurv <- ggsurvplot(
    KM_fit,                  # survfit object with calculated statistics.
    data = data,             # data used to fit survival curves.
    risk.table = "abs_pct",  # show risk table.
    xlim = c(-5, 170),
    palette = c("#8DA0CB", "#66C2A5", "#FC8D62"),
    pval = FALSE,              # show p-value of log-rank test
    test.for.trend=TRUE,
    conf.int = TRUE,           # show confidence intervals for point estimates of survival curves.
    # conf.int.style = "step",  # customize style of confidence intervals
    xlab = "Months",   # customize X axis label.
    ylab = paste(surv_name, 'probability', sep=' '),
    break.time.by = 30,     # break X axis in time intervals by 500.
    ggtheme = theme_light(), # customize plot and risk table with a theme.
    risk.table.y.text.col = T, # colour risk table text annotations.
    risk.table.height = 0.3, # the height of the risk table
    risk.table.y.text = FALSE, # show bars instead of names in text annotations
    legend.labs = c("Gleason <= 6", "Gleason 7", "Gleason >= 8")
  )
  
  ggsurv <- ggpar(
    ggsurv,
    font.title    = c(16, "bold"),               
    font.x        = c(18, "bold.italic"),          
    font.y        = c(18, "bold.italic"),      
    font.xtickslab = c(16, "plain"),
    font.ytickslab = c(16, "plain"),
    font.legend = c(16, "plain"),
    legend = "top"
  )
    
  # add the p-values
  pval1 <- pvals$p.value[1,1]
  pval2 <- pvals$p.value[2,2]
  pval3 <- pvals$p.value[2,1]
  l1 <- paste(sep = "", "low vs interm. risk: P = ", toString(round(pval1, digits=3)))
  l2 <- paste(sep = "", "interm. vs high risk: P = ", toString(round(pval2, digits=3)))
  l3 <- paste(sep = "", "low vs high risk: P = ", toString(round(pval3, digits=3)))
  ggsurv <- ggsurv + annotate("text", x=50, y=0.4, label=l1, cex=6) + 
                     annotate("text", x=50, y=0.3, label=l2, cex=6) +
                     annotate("text", x=50, y=0.2, label=l3, cex=6)
  ggsave(paste(surv_type, 'survival', annotator, 'KM.pdf', sep='_'), plot=print(ggsurv))
}

for (i in 1:3) {
  s <- choices[[i]]
  surv_name <- names[[i]]
  data <- read.csv(paste(input_directory, s, '.csv', sep=''))

  # draw survival curves for the CNN model
  fit_CNN <- survfit(Surv(time, status) ~ GL_predicted, data=data)
  # get pairwise logrank p-values, with BH correction
  pvals <- pairwise_survdiff(Surv(time, status) ~ GL_predicted, data=data, p.adjust.method="BH")
  draw_survival_curves(fit_CNN, pvals, 'CNN', s, surv_name)

  # draw survival curves based on Kim's annotations
  fit_patho1 <- survfit(Surv(time, status) ~ GL_annot_kim, data=data)
  # pairwise logrank p-values, with BH correction
  pvals <- pairwise_survdiff(Surv(time, status) ~ GL_annot_kim, data=data, p.adjust.method="BH")
  draw_survival_curves(fit_patho1, pvals, 'kim', s, surv_name)
  
  # draw survival curves based on Jan's annotations
  fit_patho2 <- survfit(Surv(time, status) ~ GL_annot_jan, data=data)
  # pairwise logrank p-values, with BH correction
  pvals <- pairwise_survdiff(Surv(time, status) ~ GL_annot_jan, data=data, p.adjust.method="BH")
  draw_survival_curves(fit_patho2, pvals, 'jan', s, surv_name)
}  

# univariate Cox regression analysis
for (i in 1:3) {
  s <- choices[[i]]
  data <- read.csv(paste(input_directory, s, '.csv', sep=''))
  print(names[[i]])
  
  fit_CNN <- coxph(Surv(time, status) ~ GL_predicted, data=data, method ="breslow")
  print(summary(fit_CNN))
  fit_patho1 <- coxph(Surv(time, status) ~ GL_annot_kim, data=data, method ="breslow")
  print(summary(fit_patho1))
  fit_patho2 <- coxph(Surv(time, status) ~ GL_annot_jan, data=data, method ="breslow")
  print(summary(fit_patho2))
}
