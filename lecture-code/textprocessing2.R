# packages = c("tm", "wordcloud", "ggplot2", "stringr", "readtext", "RmecabKo")

for(i in packages) {
  if(! require(i, character.only = TRUE)) {
    install.packages(i, dependencies = TRUE)
  }
}

# library(tm)
# library(wordcloud)
# library(ggplot2)
# library(stringr)
# library(readtext)
# library(RmecabKo)

rm(list = ls())
pdf.options(family = "Korea1deb")
pal <- brewer.pal(9, "Set1") # assigns color to graph


## Reading and Preprocessing

# document reading
folder <- "./data_edueval"
text_r <- readtext(paste0(folder, "/*.pdf"))

# preprocessing
text_r <- readtext(paste0(folder, "/*.pdf"))
text_o <- text_r$text
text_n = lapply(text_o, nouns)
txt <- text_n

# regex
txt <- gsub("@[[:graph:]]*", "", txt)
txt <- gsub("http://[[:graph:]]*", "", txt)
txt <- gsub("[^[:graph:]]", " ", txt)
txt <- gsub("[[:punct:]]", "", txt)
txt <- gsub("\n", " ", txt)
txt <- gsub("#", " ", txt)
txt <- gsub("\r", " ", txt)
txt <- gsub("RT", " ", txt)
txt <- gsub("http", " ", txt)
txt <- gsub("  ", " ", txt)

# form Corpus
corpus <- Corpus(VectorSource(txt))

# Document Term Matrix
uniTokenizer <- function(x) unlist(strsplit(as.character(x), "[[:space:]]+"))
control = list(tokenize = uniTokenizer, 
               removeNumbers = TRUE,
               wordLengths = c(4, 20),
               removePunctuation= TRUE,
               stopwords = TRUE,
               weighting = function(x) weightTfIdf(x, TRUE))
tdm <- DocumentTermMatrix(corpus, control = control)

control2 = list(tokenize = words, 
               removeNumbers = TRUE,
               wordLengths = c(4, 20),
               removePunctuation= TRUE,
               stopwords = TRUE)
tdm2 <- DocumentTermMatrix(corpus, control = control2)

# frequency analysis
TermFreq <- colSums(as.matrix(tdm2))
summary(TermFreq)

TermFreq2 <- subset(TermFreq, TermFreq > 100)
TermFreq2

gframe <- data.frame(term = names(TermFreq2), freq = TermFreq2)
ggplot(data=gframe) + aes(x = term, y = freq) + geom_bar(stat="identity") + coord_flip()

wordcloud(names(TermFreq2), TermFreq2, col=pal)

# frequency w/ weights
TermFreq <- colSums(as.matrix(tdm))
summary(TermFreq)

TermFreq2 <- NA
TermFreq2 <- subset(TermFreq, TermFreq > 0.02)
gframe <- data.frame(term = names(TermFreq2), freq = TermFreq2)
wordcloud(names(TermFreq2), TermFreq2, scale=c(4, 0.01), col=pal)

# cluster analysis
td <- removeSparseTerms(tdm, 0.99)
mdata <- as.matrix(td)
d <- dist(mdata, method = "euclidean") # distance matrix
fit <- hclust(d, method = "ward.D")
# fit <- hclust(d, method = "complete)

plot(fit)

groups <- cutree(fit, k=5)
rect.hclust(fit, k=5, border="red")
