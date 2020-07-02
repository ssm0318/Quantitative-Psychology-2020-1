#############################################
packages = c("tm", "wordcloud","ggplot2","stringr","readtext","RmecabKo","lsa",'GPArotation',"readtext","topicmodels")
for(i in packages){
  if( ! require( i , character.only = TRUE ) ){install.packages( i , dependencies = TRUE )}
}

rm(list=ls())
pal <- brewer.pal(9,"Set1")
# folder="./data_edueval"      #이 부분은 수정되어야 함#
# text_r <- readtext(paste0(folder, "/*.pdf"))
df <- read.csv("answers.csv")
df$created_at <- as.Date(df$created_at , format = "%Y-%m-%d ")

df <- df[(df[,'created_at'] >= as.Date("2019-01-28")),]
df <- df[(df[,'created_at'] <= as.Date("2019-03-31")),]

text_r = ""
for (i in 1:nrow(df)) {
  text_r <- paste(text_r, df[i, 'content'])
}

txt_n=lapply(text_r, nouns)
txt<-txt_n

txt <- gsub("@[[:graph:]]*", "",txt)
txt <- gsub("http://[[:graph:]]*", "",txt)
txt <- gsub("[^[:graph:]]", " ",txt) 
txt <- gsub("[[:punct:]]","",txt)
txt <- gsub("\n", " ", txt)
txt <- gsub("#", " ", txt)
txt <- gsub("\r", " ", txt)
txt <- gsub("RT", " ", txt)
txt <- gsub("http", " ", txt)
txt <- gsub("  ", " ",txt)
txt <- gsub("ㅋㅋ", " ",txt)
corpus <- Corpus(VectorSource(txt))

#############################################
##  Document-Term Matrix의 형성                            ##
#############################################

uniTokenizer <- function(x) unlist(strsplit(as.character(x), "[[:space:]]+"))
control1 = list(tokenize = uniTokenizer,
                removeNumbers = TRUE, 
                wordLengths=c(4,20), 
                removePunctuation = T,
                stopwords = c("그리고", "있는", '하고', "하는", "있다"),              
                weighting = function(x) weightTfIdf(x, TRUE))
tdm <- DocumentTermMatrix(corpus, control=control1)

control2 = list(tokenize = uniTokenizer, 
                removeNumbers = TRUE, 
                wordLengths=c(4,20), 
                removePunctuation = T,
                stopwords = c("그리고", "있는", '하고', "하는", "있다"))
tdm2 <- DocumentTermMatrix(corpus, control=control2)


#############################################
##                          기초 분석                              ##
#############################################

findFreqTerms(tdm2)

TermFreq <-colSums(as.matrix(tdm2))
summary(TermFreq)
TermFreq2 <- sort(subset(TermFreq, TermFreq>40))
TermFreq2

gframe<-data.frame(term = names(TermFreq2),freq = TermFreq2)
ggplot(data=gframe)+aes(x=term,y=freq)+geom_bar(stat="identity")+coord_flip()+
  theme(text = element_text(size=20))

wordcloud(names(TermFreq2), TermFreq2 , col=pal)

##########################
# Weighted DTM
##########################

findFreqTerms(tdm) 

TermFreq <-colSums(as.matrix(tdm))
summary(TermFreq)
TermFreq2<-NA
TermFreq2 <- subset(TermFreq, TermFreq>0.03)
TermFreq2

gframe<-data.frame(term = names(TermFreq2),freq = TermFreq2)
wordcloud(names(TermFreq2), TermFreq2 , col=pal)
wordcloud(names(TermFreq2), TermFreq2 ,  scale=c(4,0.01), col=pal)

##########################
# Clustering
##########################
td <- removeSparseTerms(tdm,0.99)

mdata<-as.matrix(td)
d <- dist(mdata, method = "euclidean") # distance matrix

fit <- hclust(d, method="ward.D") 

plot(fit) # display dendogram
groups <- cutree(fit, k=3) 
rect.hclust(fit, k=3, border="red")

##########################
##            LSA                    ##
##########################
LSA<-lsa(td,dim=5)
st<-LSA$tk
wd<-LSA$dk
strength<-LSA$sk

rot<-GPFoblq(wd, Tmat=diag(ncol(wd)), normalize=FALSE, eps=1e-5, maxit=10000, method="quartimin", methodArgs=NULL)

readkey <- function() {
  line <- readline(prompt="Press [enter] to continue")
}

for (i in 1:5)
{
  t<-rot$loadings[,i]
  tt<-abs(t)
  terms<-names(tt)
  COL <- ifelse(tt>0, "red", "blue")
  wordcloud(terms, tt,  scale=c(6,0.1), rot.per=0,max.words=100,random.color=T,colors=COL)
  readkey()
}

rot$Phi

td <- removeSparseTerms(tdm2,0.99)
dtm_LDA <- LDA(td, 5)
posterior<-posterior(dtm_LDA)
str(posterior)
lda_t<-data.frame(t(posterior$terms))
lda_tp<-data.frame(posterior$topics)

topics(dtm_LDA)
terms(dtm_LDA,20)

for (i in 1:5) {
  wordcloud(row.names(lda_t), lda_t[,i], max.words=50)
  readkey()
}

lda_tp$text<-rownames(lda_tp)
long<- reshape(lda_tp,varying=list(c(1,2,3,4,5)), direction='long',timevar='topic')
ggplot(long, aes(text,X1, fill=as.character(topic))) +geom_bar(stat='identity')+coord_flip()   








