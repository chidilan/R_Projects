#Twitter Airline Tweet Sentiment Analysis


#Import Airline Tweet Sentiment Dataset
library(readr)
TwtRaw <- read_csv("Tweets.csv")
str(TwtRaw)

#Convert string to timestamp
library("lubridate")
TwtRaw$tweet_created <- ymd_hms(TwtRaw$tweet_created)

#Data Screening
##Check for accuracy of quantitive variables
summary(TwtRaw)
summary(TwtRaw$airline_sentiment_confidence)
summary(TwtRaw$negativereason_confidence)

##Checking for missing data
apply(TwtRaw,2,function(x) sum(is.na(x)))

##Checking the percentage of NA or missing values
apply(TwtRaw,2,function(x) (sum(is.na(x))/length(x))*100)

##Remove all columns missing > 90% of data completely
###sentiment_gold, reason_gold do not have any data, tweet_coord is missing for 93% of the dataset and wouldn't provide us much insight
TwtRaw2 <- TwtRaw[ ,-c(7,9,12)]

##Creating a subset of customer complaints for further screening
TwtNeg <- subset(TwtRaw2, TwtRaw2$airline_sentiment == "negative")
apply(TwtNeg,2,function(x) (sum(is.na(x))/length(x))*100)

#Exploratory Data Analysis
##Summary of tweets by airline sentiment
library(dplyr)
library(reshape2)
library(ggplot2)
library("viridis")
TotTwt <- length(TwtRaw$tweet_id)
TotTwt

##Create a table of airlines & sentiments to summarize sentiment type by percentage
table(TwtRaw$airline)
table(TwtRaw$airline_sentiment)
Tab1 <- table(TwtRaw$airline, TwtRaw$airline_sentiment)
TwtTab <- as.data.frame(Tab1)
names(TwtTab)[1] <- "Airline"
names(TwtTab)[2] <- "Sentiment"
TwtTab
TwtTab2 <- dcast(TwtTab, Airline ~ Sentiment)
TwtTab3 <- mutate(TwtTab2, Total = negative + neutral + positive, Negative_Percent = (negative/Total)*100, Neutral_Percent = (neutral/Total)*100, Positive_Percent = (positive/Total)*100)
TwtTab3 <- TwtTab3 %>% mutate_if(is.numeric, round, digits = 2)
TwtTab3

##Sorting by total number of tweets
MaxTweet <- arrange(TwtTab3, desc(Total))
MaxTweet

##Sorting by decreasing order of negative % of tweets
MaxNegPercent <- arrange(TwtTab3, desc(Negative_Percent))
MaxNegPercent


##Vizualizing tweeting pattern by day
ggplot(TwtRaw, aes(x = TwtRaw$tweet_created, fill = TwtRaw$airline_sentiment, color = TwtRaw$airline_sentiment)) + geom_histogram(position = "identity", bins = 40, show.legend = TRUE) + theme_classic() + labs (x = "Date Tweeted", y = "Count") + theme(legend.position = c(0.2,0.9)) + theme(legend.title = element_blank())

##Graph of total tweets by airline
library("viridis")
barplot(height = MaxTweet$Total, names = TwtTab3$Airline, col=plasma(6), ylab = "Tweets", xlab = "Airline")

##Graph of tweets by sentiment by airline
ggplot(TwtTab, aes(fill = TwtTab$Sentiment, y = TwtTab$Freq, x =reorder(TwtTab$Airline,-TwtTab$Freq))) + geom_bar(position = "stack", stat = "identity") + scale_fill_manual("Sentiment", values = c('#D95f02','#666666','#66A61E')) + theme_classic() + labs(x = "Airline", y = "Tweet Count") + theme(legend.position = c(0.9,0.9))

##Graph of negative % of tweets
barplot(height = MaxNegPercent$Negative_Percent, names = MaxNegPercent$Airline, col=blues9, ylab = "% Negative Tweets", xlab = "Airline", ylim = c(0,100))

##Pareto of negative reason
Ntable <- table(TwtNeg$negativereason, TwtNeg$airline)
Ntab2 <- as.data.frame(Ntable)
names(Ntab2)[1] <- "Negative Reason"
names(Ntab2)[2] <- "Airline"
ggplot(Ntab2, aes(fill = Ntab2$`Negative Reason`, x = reorder(Ntab2$`Negative Reason`,Ntab2$Freq), y = Ntab2$Freq)) + geom_bar(stat = "identity") + theme_classic() + labs(x = "Negative Resaon", y = "Tweet Count") + coord_flip() + theme(legend.position = "none")

##Heatmap of airlines and negative sentiments
ggplot(Ntab2, aes(x = Ntab2$Airline, y = Ntab2$`Negative Reason`, fill = Ntab2$Freq)) + geom_tile() + scale_fill_viridis() + labs(x = "Airline", y = "Negative Reason") + labs(fill = "Tweet Count") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

##Preprocessing the text (tweet) data & tokenization
library(tidytext)
library(xml2)
library(tm)
library(wordcloud)
TwCorpus <- Corpus(VectorSource(TwtRaw$text))
TwCorpus <- tm_map(TwCorpus, content_transformer(tolower))
removeMentions <- function(x) gsub("@\\w+","",x)
TwCorpus <- tm_map(TwCorpus, removeMentions)
removeHastag <- function(x) gsub("#\\w+","",x)
TwCorpus <- tm_map(TwCorpus, removeHastag)
removeUrl <- function(x) gsub("http\\w+","",x)
TwCorpus <- tm_map(TwCorpus, removeUrl)
TwCorpus <- tm_map(TwCorpus, function(x)removePunctuation(x, preserve_intra_word_contractions = FALSE))
TwCorpus <- tm_map(TwCorpus, content_transformer(removeNumbers))
TwCorpus <- tm_map(TwCorpus, removeWords, stopwords("english"))
Clean_Text <- data.frame(text = get("content", TwCorpus))
names(Clean_Text) <- "Clean_Text"
CleanTwt <- cbind(TwtRaw,Clean_Text)

##Generating wordclouds for each type of sentiment
CleanTwtPos <- subset(CleanTwt, CleanTwt$airline_sentiment == "positive")
CleanTwtNeutral <- subset(CleanTwt, CleanTwt$airline_sentiment == "neutral")
CleanTwtNeg <- subset(CleanTwt, CleanTwt$airline_sentiment == "negative")

##Corpus Analysis
###unigram pareto
Clean_Text <- data.frame(txt = CleanTwtNeg$Clean_Text, stringsAsFactors = FALSE)
Clean_Text %>% unnest_tokens(Issues, txt, token = "ngrams", n = 1) %>% count(Issues, sort = TRUE)

###bigram pareto
Clean_Text <- data.frame(txt = CleanTwtNeg$Clean_Text, stringsAsFactors = FALSE)
Clean_Text %>% unnest_tokens(Issues, txt, token = "ngrams", n = 2) %>% count(Issues, sort = TRUE)

###word cloud for negative reasons
Neg_tdm <- as.matrix(TermDocumentMatrix(Corpus(VectorSource(CleanTwtNeg$Clean_Text))))
Neg_Order <- sort(rowSums(Neg_tdm), decreasing = TRUE)
Neg_dfm <- data.frame(Nword = names(Neg_Order), freq=Neg_Order)
wordcloud(Neg_dfm$Nword, Neg_dfm$freq, min.freq = 85, colors = brewer.pal(8, "Dark2"))

###word cloud for positive reasons
Pos_tdm <- as.matrix(TermDocumentMatrix(Corpus(VectorSource(CleanTwtPos$Clean_Text))))
Pos_Order <- sort(rowSums(Pos_tdm), decreasing = TRUE)
Pos_dfm <- data.frame(Nword = names(Pos_Order), freq=Pos_Order)
wordcloud(Pos_dfm$Nword, Pos_dfm$freq, min.freq = 40, colors = brewer.pal(8, "Dark2"))

##word frequency by airline and sentiment
library(dplyr)
library(tidytext)
library(stringr)

###create subsets of negative sentiments by airlines
Southwest <- data.frame(txt = CleanTwtNeg$Clean_Text[CleanTwtNeg$airline == "Southwest"], stringsAsFactors = FALSE)
Delta <- data.frame(txt = CleanTwtNeg$Clean_Text[CleanTwtNeg$airline == "Delta"], stringsAsFactors = FALSE)
United <- data.frame(txt = CleanTwtNeg$Clean_Text[CleanTwtNeg$airline == "United"], stringsAsFactors = FALSE)
American <- data.frame(txt = CleanTwtNeg$Clean_Text[CleanTwtNeg$airline == "American"], stringsAsFactors = FALSE)
Virgin <- data.frame(txt = CleanTwtNeg$Clean_Text[CleanTwtNeg$airline == "Virgin America"], stringsAsFactors = FALSE)

###generate word counts by airlines
Southwest <- Southwest %>% unnest_tokens(words, txt, token = "ngrams", n = 1) %>% count(words, sort = TRUE)
delta <- Delta %>% unnest_tokens(words, txt, token = "ngrams", n = 1) %>% count(words, sort = TRUE)
United <- United %>% unnest_tokens(words, txt, token = "ngrams", n = 1) %>% count(words, sort = TRUE)
American <- American %>% unnest_tokens(words, txt, token = "ngrams", n = 1) %>% count(words, sort = TRUE)
Virgin <- Virgin %>% unnest_tokens(words, txt, token = "ngrams", n = 1) %>% count(words, sort = TRUE)

wordFreq <- bind_rows(mutate(Southwest, airline = "Southwest"),mutate(Delta, airline = "Delta"), mutate(United, airline = "United"), mutate(American, airline = "American"), mutate(Virgin, airline = "Virgin America")) %>% group_by(airline) %>% mutate (ratio = n / sum(n)) %>% select(-n)

wordFreq <- bind_rows(mutate(Southwest, airline = "Southwest"),mutate(Delta, airline = "Delta"), mutate(United, airline = "United"), mutate(American, airline = "American"), mutate(Virgin, airline = "Virgin America")) %>% group_by(airline) %>% mutate (ratio = n / sum(n)) %>% select(-n) %>% pivot_wider(names_from = airline, values_from = ratio, values_fill = list(ratio = 0)) %>% pivot_longer(3:6, names_to = "Others", values_to = "ratio")

wordFreq <- bind_rows(mutate(Southwest, airline = "Southwest"),mutate(Delta, airline = "Delta"), mutate(United, airline = "United"), mutate(American, airline = "American")) %>% group_by(airline) %>% mutate (ratio = n / sum(n)) %>% select(-n) %>% pivot_wider(names_from = airline, values_from = ratio, values_fill = list(ratio = 0)) %>% pivot_longer(3:5, names_to = "Others", values_to = "ratio")

#Hypothesis testig using word frequency correlations (using PPMC)
cor.test(data = wordFreq[wordFreq$Others == "United",], ~ ratio + `Southwest`)
cor.test(data = wordFreq[wordFreq$Others == "Delta",], ~ ratio + `Southwest`)
cor.test(data = wordFreq[wordFreq$Others == "American",], ~ ratio + `Southwest`)
cor.test(data = wordFreq[wordFreq$Others == "Virgin America",], ~ ratio + `Southwest`)

#######################
