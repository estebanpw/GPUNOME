
#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
 
if(length(args) < 1){
	stop("USE: Rscript --vanilla plot_and_score.R <matrix>")
}

path_mat = args[1]

fancy_name <- strsplit(path_mat, "/")
fancy_name <- (fancy_name[[1]])[length(fancy_name[[1]])]


data <- as.matrix(read.csv(path_mat, sep = " "))

score_copy <- data

len_i <- 1000

# To compute the score
score <- 0
pmax_pos <- which.max(score_copy[,1])
dist_th <- 1.5
besti <- 1
bestj <- pmax_pos
dvec1 <- abs(which.max(score_copy[,2]) - which.max(score_copy[,1]))
dvec2 <- abs(which.max(score_copy[,3]) - which.max(score_copy[,2]))
dvec3 <- abs(which.max(score_copy[,4]) - which.max(score_copy[,3]))
for(i in 5:len_i){
  
  #print(paste(paste(paste(dvec1, dvec2), dvec3), mean(c(dvec1, dvec2, dvec3))))
  distance <- mean(c(dvec1, dvec2, dvec3))
  
  
  
  dvec1 <- dvec2
  dvec2 <- dvec3
  dvec3 <- abs(which.max(score_copy[,i]) - which.max(score_copy[,i-1]))
  
  # If there is a 0 or we are too far away just add max distance!
  if(distance > dist_th || distance == 0){
    score <- score + len_i
  }
  
}


score <- (score/(len_i^2))
print(score)

png(paste(path_mat, ".filt.png", sep=""), width = length(data[,1]), height = length(data[,1]))
image(t(score_copy), col = grey(seq(1, 0, length = 256)), xaxt='n', yaxt='n', main = paste(fancy_name, paste("filt. score=", score)))
dev.off()

