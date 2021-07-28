library(ggplot2)
library(RColorBrewer)
library(dplyr)
user_path_computingtime = "YOUR_PATH"
setwd(user_path_computingtime)

comp = rep(0,7)
for(i in 1:4){
  comptemp = read.csv(paste("50-6-4-10",i,"2400-BIC-COMP.csv",sep="-"))
  comp = cbind(comp,comptemp[1:7,2])
}
comp = comp[,2:5]
comp
rowMeans(comp)

plot(1:7,rowMeans(comp))

data = data.frame(time = as.vector(comp),
                  kp = rep(1:7,4),
                  group = rep("A",4*7))
mean_data <- group_by(data, kp) %>%
  summarise(mean = mean(time, na.rm = TRUE),
            SE = sd(time, na.rm = TRUE)/sqrt(4))
mean_data$group = "A"
l = 1.5
w = 0.5
fnt = 18
a = 0.2
fnt2=12
pall = ggplot(mean_data, aes(x=kp,y=mean,colour = group,group=group)) +
  geom_line(size=l,aes(color="#252525")) +
  theme_bw() +
 geom_errorbar(aes(ymin=mean-SE, ymax=mean+SE,color = "#252525"), width=w) +
  theme(axis.text.x=element_text(size=fnt,face="bold"),axis.text.y=element_text(size=fnt,face="bold")) +
  theme(axis.title.x = element_text(size=fnt,face="bold")) + 
  theme(axis.title.y = element_text(size=fnt,face="bold")) +
   theme(strip.text.x = element_text(size = fnt2,face="bold")) +
  ylab("Computing time (seconds)") +
  xlab("Feature sparsity level") + 
  theme(legend.title = element_blank())+
  theme(legend.text=element_text(size=fnt2,face="bold")) +
  scale_color_manual(values="#525252") +
  theme(legend.position = "none")
pall
