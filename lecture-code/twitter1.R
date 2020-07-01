
library(rtweet)
#library(curl)
library(tm)
library(ggplot2)
library(wordcloud)
library(RmecabKo)

pal<-brewer.pal(9,"Set1")

## access token method: create token and save it as an environment variable
create_token(
  app = "이 부분을 채워넣어야 함",
  consumer_key = "이 부분을 채워넣어야 함",
  consumer_secret = "이 부분을 채워넣어야 함",
  access_token = "이 부분을 채워넣어야 함",
  access_secret = "이 부분을 채워넣어야 함")


twtxt <- search_tweets("#코로나",n=50000,lang="ko")

#########################################################################

dim(twtxt)

txt<-twtxt$text

txt <- gsub("@[[:graph:]]*", "",txt)
txt <- gsub("http://[[:graph:]]*", "",txt)
txt <- gsub("[^[:graph:]]", " ",txt) 
txt <- gsub("[[:punct:]]","",txt)
txt <- gsub("\n", " ", txt)
txt <- gsub("#", "", txt)
txt <- gsub("\r", "", txt)
txt <- gsub("RT", "", txt)
txt <- gsub("http", "", txt)
txt <- gsub("CO", "", txt)
txt <- gsub("co", "", txt)
txt <- gsub("ㅋㅋ", "", txt)
txt <- gsub("ㅋㅋㅋ", "", txt)
txt <- gsub("ㅠㅠ", "", txt)


#txt_n=nouns(txt)

txt_n=lapply(txt,nouns)

corpus <- Corpus(VectorSource(txt_n))

#############################################
##  Document Term Matrix의 형성            ##
#############################################
uniTokenizer <- function(x) unlist(strsplit(as.character(x), "[[:space:]]+"))
control = list(tokenize = uniTokenizer, 
               removeNumbers = FALSE, 
               wordLengths=c(2,20), 
               removePunctuation = T,
               stopwords = c("\\n","\n","것"),
               weighting = function(x) weightTfIdf(x, FALSE))
tdm <- DocumentTermMatrix(corpus, control=control)
td<-tdm



#############################################
##  기초 분석                              ##
#############################################

t<-findFreqTerms(td, lowfreq=500)
iconv(t, "CP949", "UTF-8")

TermFreq <-colSums(as.matrix(tdm))
TermFreq2 <- subset(TermFreq, TermFreq>500)
nhan<-iconv(names(TermFreq2), "CP949", "UTF-8")
gframe<-data.frame(term = nhan,freq = TermFreq2)
ggplot(data=gframe)+aes(x=term,y=freq)+geom_bar(stat="identity")+coord_flip()
wordcloud(nhan, TermFreq2 ,  scale=c(3,0.1),colors=brewer.pal(6, "Dark2"))











