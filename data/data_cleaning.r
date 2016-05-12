#install.packages("jsonlite")
#install.packages("splitstackshape")

library(jsonlite)
library(plyr)
library(tidyr)
library(dplyr)
library(splitstackshape)

#Getting the data from train.json file
json_file <- "C:\\Users\\ggangwan\\Documents\\train.json"
json_data <- fromJSON(json_file)

unique(json_data$cuisine)


#Separating ingredients based on their cuisine
greek         <- json_data[grep("greek", json_data$cuisine), ]
southern_us   <- json_data[grep("southern_us", json_data$cuisine), ]
filipino      <- json_data[grep("filipino", json_data$cuisine), ]
indian        <- json_data[grep("indian", json_data$cuisine), ]
jamaican      <- json_data[grep("jamaican", json_data$cuisine), ]
spanish       <- json_data[grep("spanish", json_data$cuisine), ]
italian       <- json_data[grep("italian", json_data$cuisine), ]
mexican       <- json_data[grep("mexican", json_data$cuisine), ]
chinese       <- json_data[grep("chinese", json_data$cuisine), ]
british       <- json_data[grep("british", json_data$cuisine), ]
thai          <- json_data[grep("thai", json_data$cuisine), ]
vietnamese    <- json_data[grep("vietnamese", json_data$cuisine), ]
cajun_creole  <- json_data[grep("cajun_creole", json_data$cuisine), ]
brazilian     <- json_data[grep("brazilian", json_data$cuisine), ]
french        <- json_data[grep("french", json_data$cuisine), ]
japanese      <- json_data[grep("japanese", json_data$cuisine), ]
irish         <- json_data[grep("irish", json_data$cuisine), ]
korean        <- json_data[grep("korean", json_data$cuisine), ]
moroccan      <- json_data[grep("moroccan", json_data$cuisine), ]
russian       <- json_data[grep("russian", json_data$cuisine), ]

#Unlisting the elements of ingredients
greek_ingredients       <- unlist(greek$ingredients)
cuisine <- c("greek")
list_greek <- cbind(cuisine, greek_ingredients)
colnames(list_greek)[2] <- "ingredients"
southern_us_ingredients <- unlist(southern_us$ingredients)
cuisine <- c("southern_us")
list_southern_us <- cbind(cuisine, southern_us_ingredients)
colnames(list_southern_us)[2] <- "ingredients"
filipino_ingredients    <- unlist(filipino$ingredients)
cuisine <- c("filipino")
list_filipino <- cbind(cuisine, filipino_ingredients)
colnames(list_filipino)[2] <- "ingredients"
indian_ingredients      <- unlist(indian$ingredients)
cuisine <- c("indian")
list_indian <- cbind(cuisine, indian_ingredients)
colnames(list_indian)[2] <- "ingredients"
jamaican_ingredients    <- unlist(jamaican$ingredients)
cuisine <- c("jamaican")
list_jamaican <- cbind(cuisine, jamaican_ingredients)
colnames(list_jamaican)[2] <- "ingredients"
spanish_ingredients     <- unlist(spanish$ingredients)
cuisine <- c("spanish")
list_spanish <- cbind(cuisine, spanish_ingredients)
colnames(list_spanish)[2] <- "ingredients"
italian_ingredients     <- unlist(italian$ingredients)
cuisine <- c("italian")
list_italian <- cbind(cuisine, italian_ingredients)
colnames(list_italian)[2] <- "ingredients"
mexican_ingredients     <- unlist(mexican$ingredients)
cuisine <- c("mexican")
list_mexican <- cbind(cuisine, mexican_ingredients)
colnames(list_mexican)[2] <- "ingredients"
chinese_ingredients     <- unlist(chinese$ingredients)
cuisine <- c("chinese")
list_chinese <- cbind(cuisine, chinese_ingredients)
colnames(list_chinese)[2] <- "ingredients"
british_ingredients     <- unlist(british$ingredients)
cuisine <- c("british")
list_british <- cbind(cuisine, british_ingredients)
colnames(list_british)[2] <- "ingredients"
thai_ingredients        <- unlist(thai$ingredients)
cuisine <- c("thai")
list_thai <- cbind(cuisine, thai_ingredients)
colnames(list_thai)[2] <- "ingredients"
vietnamese_ingredients  <- unlist(vietnamese$ingredients)
cuisine <- c("vietnamese")
list_vietnamese <- cbind(cuisine, vietnamese_ingredients)
colnames(list_vietnamese)[2] <- "ingredients"
cajun_creole_ingredients<- unlist(cajun_creole$ingredients)
cuisine <- c("cajun_creole")
list_cajun_creole <- cbind(cuisine, cajun_creole_ingredients)
colnames(list_cajun_creole)[2] <- "ingredients"
brazilian_ingredients   <- unlist(brazilian$ingredients)
cuisine <- c("brazilian")
list_brazilian <- cbind(cuisine, brazilian_ingredients)
colnames(list_brazilian)[2] <- "ingredients"
french_ingredients      <- unlist(french$ingredients)
cuisine <- c("french")
list_french <- cbind(cuisine, french_ingredients)
colnames(list_french)[2] <- "ingredients"
japanese_ingredients    <- unlist(japanese$ingredients)
cuisine <- c("japanese")
list_japanese <- cbind(cuisine, japanese_ingredients)
colnames(list_japanese)[2] <- "ingredients"
irish_ingredients       <- unlist(irish$ingredients)
cuisine <- c("irish")
list_irish <- cbind(cuisine, irish_ingredients)
colnames(list_irish)[2] <- "ingredients"
korean_ingredients      <- unlist(korean$ingredients)
cuisine <- c("korean")
list_korean <- cbind(cuisine, korean_ingredients)
colnames(list_korean)[2] <- "ingredients"
moroccan_ingredients    <- unlist(moroccan$ingredients)
cuisine <- c("moroccan")
list_moroccan <- cbind(cuisine, moroccan_ingredients)
colnames(list_moroccan)[2] <- "ingredients"
russian_ingredients     <- unlist(russian$ingredients)
cuisine <- c("russian")
list_russian <- cbind(cuisine, russian_ingredients)
colnames(list_russian)[2] <- "ingredients"

df <- rbind(list_greek, list_southern_us, list_filipino, list_indian, list_jamaican, list_spanish, list_italian, list_mexican, list_chinese, list_british, list_thai, list_vietnamese, list_cajun_creole, list_brazilian, 
list_french, list_japanese, list_irish, list_korean, list_moroccan, list_russian)

