#
# This is a Shiny web application. The comprehensive report is in
# https://github.com/xuz057/MA615_Group_Project/tree/master/Final_Project

# install.packages("tidyverse")

library(shiny)
library(foreign)
library(shinydashboard)
library(tidyverse)
library(magrittr)
library(readr)
library(scales)
library(knitr)
library(benford.analysis)
library(BenfordTests)
library(DT)
library(plotly)
library(tidyr)
options("scipen"=100, "digits"=4)

#########


ui <- dashboardPage( 
  dashboardHeader(title = "Using Benford's Law to Study the Current Ratio of Corporate Bankruptcy",
                  titleWidth = 700
  ),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Hello!", tabName = "hello",icon = icon("coffee")),
      menuItem("Benford Analysis", tabName = "BA",icon = icon("chart-line")),
      menuItem("Source Data", tabName = "SD", icon = icon("clipboard-list"))
    )
  ),
  
  dashboardBody(
    # Also add some custom CSS to make the title background area the same
    # color as the rest of the header.
    tags$head(tags$style(HTML('
                              .skin-purple .main-header .logo {
                              background-color: #3c8dbc;
                              }
                              .skin-purple .main-header .logo:hover {
                              background-color: #3c8dbc;
                              }
              '))),
    tabItems(
      tabItem(tabName = "hello",
              fluidRow(
                box(
                  title = "To the Readers",
                  width = 12,
                  solidHeader = T,
                  status = "primary",
                  collapsible = T,
                  print("The current ratio is one of the indicators that analysts use to predict corporate bankruptcy in a financial forecasting period. In this project, Benford's law is applied to explore whether the distribution of the first leading digits has a pattern that helps analysts be more alert to unusual reports.")
                ),
                box(
                  title = "What is the Current Ratio?",
                  width = 12,
                  solidHeader = T,
                  status = "warning",
                  collapsible = T,
                  print("The current ratio is a term often used in the area of financial analysis. It measures one company's ability to pay short-term liability and long-term obligations. To calculate the ratio, analysts apply the following formula:"),
                  h4("Current Ratio = Current Assets / Current Liabilities"),
                  print("The current assets include the assets that can be transformed into cash in less than a year, such as cash and inventory. The current liabilities are how much the company needs to pay to other subjects in the future, such as taxes payable and wages.It is hard to say how good or bad a current ratio is, as the standard varies a lot across industries. To avoid this, we only select out the data of companies within the same industry (Polish companies in the manufacturing sector, which will be discussed with details later in Chapter Source Data.) But even in the same industry, getting a current ratio above the industry average line is not the same as the signal of well management, because it is possible that the company resources are not efficiently used. Similarly, getting a current ratio below one is typically an alarm that now the company is not able to pay all its obligations if they are all due at one, but the company may be struggling in one financial reporting period and then becomes healthy in the future. In a short summary, the current ratio itself is limited in comparison between different companies without other liquidity ratios' help, and we should always look at the trending information rather than focusing on analyzing data from a single financial period. ")
                  
                ),
                box(
                  title = "What is Benford's Law?",
                  width = 12,
                  solidHeader = T,
                  status = "success",
                  collapsible = T,
                  print("Benford's Law is an observation about the frequency distribution of leading digits in any large, randomly produced data sets of numbers."),
                  plotlyOutput("benford"),
                  print("In reality, the Benford's Law is widely applied to many areas such as detecting fraud. After testing whether one data set is consistent with the law, analysts narrow down their search on numbers with suspect leading digits that may be manipulated manually. In this project, we are focusing on the First Digit Benford Test, and we expect that the current ratio reported by those companies going bankrupt in five years is manipulated and thus the distribution of first leading digits deviates from the law.")
                )
                
              )#fluid
      ),
      tabItem(
        tabName = "BA",
        fluidRow(
          box(
            title = "Choose a year",
            width = 4,
            solidHeader = T,
            status = "info",
            collapsible = T,
            collapsed = F,
            selectInput(
              "year",
              "Choose one Year",
              choice = c("First","Second","Third","Fourth","Fifth"),
              selected = "First"
            )
          ),
          box(
            title = "Ideas",
            width = 8,
            solidHeader = T,
            status = "primary",
            collapsible = T,
            print("The first step is to count the frequencies of the first leading digits from 1 to 9 and calculate the proportion of each digit in the data from the frequencies. Then we compare the actual first digit frequency distribution with the theoretical Benford's one by chi square test and visualize it.")
          ),
          box(
            title = "Bankruptcy",
            width = 6,
            solidHeader = T,
            status = "warning",
            collapsible = T,
            plotlyOutput("plotb")),
          box(
            title = "Healthy",
            width = 6,
            solidHeader = T,
            status = "success",
            collapsible = T,
            plotlyOutput("ploth")),
          box(
            title = "Wanna view trending information?",
            width = 12,
            solidHeader = T,
            status = "info",
            collapsible = T,
            collapsed = F,
            selectInput(
              "type",
              "Bankrupt?Healthy?",
              choice = c("Bankrupt","Healthy"),
              selected = "Bankrupt"
            )
          ),
          box(
            title = "Trending info plots",
            h3("1st Year"),
            width = 4,
            
            solidHeader = T,
            status = "warning",
            collapsible = F,
            plotlyOutput("plotall1")
          ),
          box(
            title = "Trending info plots",
            h3("2nd Year"),
            width = 4,
            
            solidHeader = T,
            status = "warning",
            collapsible = F,
            plotlyOutput("plotall2")
          ),
          box(
            title = "Trending info plots",
            h3("3rd Year"),
            width = 4,
            
            solidHeader = T,
            status = "warning",
            collapsible = F,
            plotlyOutput("plotall3")
          ),
          box(
            title = "Trending info plots",
            h3("4th Year"),
            width = 4,
            solidHeader = T,
            status = "warning",
            collapsible = F,
            plotlyOutput("plotall4")
          ),
          box(
            title = "Trending info plots",
            h3("5th Year"),
            width = 4,
            solidHeader = T,
            status = "warning",
            collapsible = F,
            plotlyOutput("plotall5")
          )
          
        )#Fluidrow
      ),#tabitem
      
      tabItem(
        tabName = "SD",
        fluidRow(
          box(
            title = "SOURCE DATA",
            width = 12,
            status = "info",
            collapsible = T,
            print("According to UCI Machine Learning Repository, the data is first extracted and used by Zieba, M., Tomczak, S. K., and Tomczak, J. M. (2016). This chapter describes the basic information about the data set that is related to the project.

The dataset contains five subsets, each has 64 indicators predicting corporate bankruptcy from the first year of forecasting period to the fifth year of it. The companies classified as 0 in the dataset remained healthy financial status after the forecasting period, while those classified as 1 did not survive successfully. The explanation of 64 indicators is appended in the Appendix chapter. In the project, I extract out the column of X4/Attr 4 as the subject of the study. 

Two things about the quality of the data set need to be mentioned. First, the data is heavily imbalanced. There are much more healthy companies existing than the problematic ones, as you can see in the above table. Second, the data is not comprehensive about all the companies operating in the forecasting period, because some information is not public or missing from the original database.")
          )
        ),
        fluidRow(
          tabBox(
            title = "1st-5th year of a forecasting period",
            width = 12,
            height = "500px",
            tabPanel("FirstYear",  dataTableOutput("firstyear")),
            tabPanel("SecondYear", dataTableOutput("secondyear")),
            tabPanel("ThirdYear", dataTableOutput("thirdyear")),
            tabPanel("FourthYear", dataTableOutput("fourthyear")),
            tabPanel("FifthYear", dataTableOutput("fifthyear")))
        )
      )#tabitem
      
    )#tabitems
    
  ))#dashbody

#dashoboardpage

#########################

server <- function(input, output) {
  #load data sets
  oneyear <- read.arff("1year.arff") %>% select(Attr4,class)
  twoyear <- read.arff("2year.arff") %>% select(Attr4,class)
  threeyear <- read.arff("3year.arff") %>% select(Attr4,class)
  fouryear <- read.arff("4year.arff") %>% select(Attr4,class)
  fiveyear <- read.arff("5year.arff") %>% select(Attr4,class)
  #rename colnumns
  colnames(oneyear) <- c("CA_SL","class")
  colnames(twoyear) <- c("CA_SL","class")
  colnames(threeyear) <- c("CA_SL","class")
  colnames(fouryear) <- c("CA_SL","class")
  colnames(fiveyear) <- c("CA_SL","class")
  fouryear %>% filter(CA_SL<0)
  fiveyear %>% filter(CA_SL<0)
  fouryear <- fouryear %>% filter(CA_SL>0)
  fiveyear <- fiveyear %>% filter(CA_SL>0)
  
  # The Standard Distribution
  Benford <- c(0.3010,0.1761,0.1249,0.0969,0.079,0.0670,0.0580,0.0512,0.0458)
  # convert numeric values to percentages
  percent <- function(x, digits = 2, format = "f") {
    paste0(formatC(100*x, format = format, digits = digits), "%")
  }
  # produce proportion tables & plots for each data set
  Analysis_b <- function(data){ #for problematic company
    b1 <- data%>%filter(class=="1")
    result <-freq(b1)
    return(result)
  }
  
  Analysis_h <- function(data){ #for healthy company
    b1 <- data%>%filter(class=="0")
    result <-freq(b1)
    return(result)
  }
  # frequency table
  freq <- function(data){
    data$Digit <- signifd(data$CA_SL)
    freqtable <- data%>%group_by(Digit)%>%summarise(freq=n())
    freqtable <- data.frame(freqtable[c(1:9),])
    freqtable$proportion <- freqtable$freq/sum(freqtable$freq)
    freqtable$Benford <- Benford
    return(freqtable)
  }
  #print good-looking numbers in percentages
  printfunction <- function(data){
    data$proportion <- percent(data$proportion)
    data$Benford <- percent(data$Benford)
    return(data)
  }
  
  ID <- 1:9
  
  #ggplot function
  plotf <- function(data){
    data1 <- data %>% gather(Benford,proportion,key="type",value="numbers")
    plotbase <- ggplot(data1,aes(x=Digit,y=numbers,group=type,color=type))+geom_line(size=1.5)+geom_point(size=2)+scale_x_continuous("Digits", labels = as.character(ID), breaks = ID)+labs(y="Proportion",title="First Digit Test Proportions")+ theme(panel.background = element_blank(),axis.line = element_line(colour = "black"))+geom_abline(slope=0)
    return(plotbase)
  }
  #sample command
  #plotf(Analysis_b(fiveyear))
  
  ##Hello tab
  output$benford <- renderPlotly(
    {  
      numbers <- c(1,2,3,4,5,6,7,8,9)
      Benfordsample <- cbind.data.frame(numbers,Benford)
      ggplot(Benfordsample,aes(y=Benford,x=numbers))+geom_point(color="red")+geom_line(color="red")+ 
        scale_x_continuous("Digits", labels = as.character(ID), breaks = ID)+theme(panel.background = element_blank(),plot.background = element_rect(size=0.2,linetype="solid",color="black"))+ggtitle("Benford's Law Visualization")+labs(y="Frequency Distribution")
      
    }
  )
  
  ## Benford tab
  output$plotb <- renderPlotly({
    if (input$year == "First"){
      plotf(Analysis_b(oneyear))
    }else if (input$year == "Second"){
      plotf(Analysis_b(twoyear))
    }else if (input$year == "Third"){
      plotf(Analysis_b(threeyear))
    }else if (input$year == "Fourth"){
      plotf(Analysis_b(fouryear))
    }else if (input$year == "Fifth"){
      plotf(Analysis_b(fiveyear))
    }
  })
  
  output$ploth <- renderPlotly({
    if (input$year == "First"){
      plotf(Analysis_h(oneyear))
    }else if (input$year == "Second"){
      plotf(Analysis_h(twoyear))
    }else if (input$year == "Third"){
      plotf(Analysis_h(threeyear))
    }else if (input$year == "Fourth"){
      plotf(Analysis_h(fouryear))
    }else if (input$year == "Fifth"){
      plotf(Analysis_h(fiveyear))
    }
  } )
  
  plotf1 <- function(data){
    data1 <- data %>% gather(Benford,proportion,key="type",value="numbers")
    plotbase <- ggplot(data1,aes(x=Digit,y=numbers,group=type,color=type))+geom_line(size=1.5)+geom_point(size=2)+scale_x_continuous("Digits", labels = as.character(ID), breaks = ID)+labs(y="Proportion")+ theme(panel.background = element_blank(),axis.line = element_line(colour = "black"),legend.position="none")
    return(plotbase)
  }
  p1 <- plotf1(Analysis_b(oneyear))
  p2 <- plotf1(Analysis_h(oneyear))
  p3 <- plotf1(Analysis_b(twoyear))
  p4 <- plotf1(Analysis_h(twoyear))
  p5 <- plotf1(Analysis_b(threeyear))
  p6 <- plotf1(Analysis_h(threeyear))
  p7 <- plotf1(Analysis_b(fouryear))
  p8 <- plotf1(Analysis_h(fouryear))
  p9 <- plotf1(Analysis_b(fiveyear))
  p10 <-plotf1(Analysis_h(fiveyear))
  
  output$plotall1 <- renderPlotly({
    if (input$type =="Bankrupt"){
      p1
    }else{p2}
  })
  output$plotall2 <- renderPlotly({
    if (input$type =="Bankrupt"){
      p3
    }else{p4}
  })
  output$plotall3 <- renderPlotly({
    if (input$type =="Bankrupt"){
      p5
    }else{p6}
  })
  output$plotall4 <- renderPlotly({
    if (input$type =="Bankrupt"){
      p7
    }else{p8}
  })
  output$plotall5 <- renderPlotly({
    if (input$type =="Bankrupt"){
      p9
    }else{p10}
  })
  
  ##Data set
  
  output$firstyear <- renderDataTable({
    DT::datatable(oneyear, options = list(lengthMenu = c(5, 10,15), pageLength = 5))
  })
  output$secondyear <- renderDataTable({
    DT::datatable(twoyear, options = list(lengthMenu = c(5, 10,15), pageLength = 5))
  })
  
  output$thirdyear <- renderDataTable({
    DT::datatable(threeyear, options = list(lengthMenu = c(5, 10,15), pageLength = 5))
  })
  output$fourthyear <- renderDataTable({
    DT::datatable(fouryear, options = list(lengthMenu = c(5, 10,15), pageLength = 5))
  })
  output$fifthyear <- renderDataTable({
    DT::datatable(fiveyear, options = list(lengthMenu = c(5, 10,15), pageLength = 5))
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)
