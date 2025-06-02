# Shiny {#shiny}

**Shiny**是R中用于构建网页应用程序的一个框架，能够为用户呈现交互式数据分析结果。无需HTML、CSS或者JavaScript等知识，你就可以基于R代码搭建出任何人都可见的数据可视化网页。

本章节内容基于[Mastering Shiny](https://mastering-shiny.org/index.html)。

同时大家也可以参考[shiny官网](https://shiny.posit.co/r/components/)了解更多细节。

## 初始Shiny {#shiny_1}

下面是一段demo。


``` r
library(shiny)
ui <- fluidPage(
  selectInput("dataset", label = "Dataset", choices = ls("package:datasets")),
  verbatimTextOutput("summary"),
  tableOutput("table")
)
server <- function(input, output, session) {
  # Create a reactive expression
  dataset <- reactive({
    get(input$dataset, "package:datasets")
  })

  output$summary <- renderPrint({
    # Use a reactive expression by calling it like a function
    summary(dataset())
  })
  
  output$table <- renderTable({
    dataset()
  })
}
shinyApp(ui, server)
```

由四部分组成：导包、`ui`、`server`、以及`shinyApp`。

其中`ui`用于网页的ui设计，`server`用于背后的数据分析，`shinyApp`用于搭建shiny应用。

那么该如何创建shiny文件呢？

1. 新建脚本文件，完成代码后保存文件，脚本文件会自动识别出这是一个shiny应用，此时你的工具栏中的`Run`按钮会变为`Run App`

2. File -> New File -> Shiny Web App

3. File -> New Project -> New Directory -> Shiny Application

至此，你可点击`Run App`按钮运行文件，或者使用`Ctrl+Shift+Enter`运行。

注意到，当你运行shiny应用时会弹出一个窗口，并且控制台处会出现类似`Listening on http://127.0.0.1:4674`的反馈。**你可以直接将该URL复制到你的网页浏览器中，这样别人也可以看到你的应用，当然前提是你的shiny应用仍在运行中**。

shiny应用是一直在运行的，你可以通过控制台右上角的红色停止按钮判断。当你点击该按钮，或者关闭shiny应用窗口时，方可停止。在停止之前，你不能在控制台中执行任何新命令。

## UI设计 {#shiny_2}

更多的输入与输出组件参见[shiny官网](https://shiny.posit.co/r/components/)。

### 输入 {#shiny_2_1}

```
sliderInput("min", "Limit (minimum)", value = 50, min = 0, max = 100)
```

对于形如`selectInput()`的函数，都是将你的信息传递给server的组件。这类组件的基本参数为ID、标签及其余参数。其中ID是组件的唯一标识符，用于后续调用（这里的ID是`min`，后续在server函数中通过`input$min`来调用该组件的输入值）。标签是该组件呈现给用户时所用的标签（这里的标签为`Limit (minimum)`，也就是页面中滑条的标签）。其余参数往往是与组件类型相关的参数（这里`value=50`表示默认值，`min`与`max`则是滑条的上下限）。


``` r
ui <- fluidPage(
  textInput("name", "What's your name?"),
  passwordInput("password", "What's your password?"),
  textAreaInput("story", "Tell me about yourself", rows = 3)
)
```

#### 文本类组件 {#shiny_2_1_1}

1. textInput()

   适合单行文本输入

2. passwordInput()

   适合密码类文本输入

3. textAreaInput()

   适合多行文本输入

#### 数值类组件 {#shiny_2_1_2}

1. numericInput()

   输入单个数值

2. sliderInput()

   滑块选取数值。特别的，当默认值是一个长度为2的向量时，滑块条变成范围取值（双向），而非单个数值（单向）
   
#### 日期类组件 {#shiny_2_1_3}

1. dateInput()

   输入日期

2. dateRangeInput()

   输入日期范围
   
#### 选择类组件 {#shiny_2_1_4}

1. selectInput()

   下拉式选择，可多选
   
2. selectizeInput()

   `selectInput()`的增强版，适合更为复杂、大量选择的输入场景

3. radioButtons()

   按钮式选择，只能单选
   
4. checkboxGroupInput()

   勾选式选择，可单选可多选
   
5. checkboxInput()

   勾选式选择，只能单选

> 注意参数`choiceNames`表示呈现给用户的选项名称，参数`choiceValues`表示传递给服务器端的实际值，二者一一对应

#### 文件传输类组件 {#shiny_2_1_5}

1. fileInput()

   将用户的文件传输到服务器端

#### 按钮类组件 {#shiny_2_1_6}

1. actionButton()

   通过点击传输信息。其中可根据`class`参数更改按钮的外观，详见[此处](https://bootstrapdocs.com/v3.3.6/docs/css/#buttons)

### 输出 {#shiny_2_2}

```
textOutput("text")
```

对于形如`textOutput()`的函数，用于接收服务器输出的结果。同理，输出的内容需要由唯一标识符ID来进行匹配。当输出的结果在服务器端中用`output$ID`完成赋值，在ui中的输出函数中即可输入相应的ID来调用结果。


``` r
ui <- fluidPage(
  textOutput("text"),
  verbatimTextOutput("code")
)
server <- function(input, output, session) {
  output$text <- renderText({ 
    "Hello friend!" 
  })
  output$code <- renderPrint({ 
    summary(1:10) 
  })
}
```

#### 文本类输出 {#shiny_2_2_1}

1. textOutput()

   用于一般的文本输出，常与渲染函数`renderText()`搭配

2. verbatimTextOutput()

   用于代码结果的输出，常与渲染函数`renderPrint()`搭配

> 注意到，如果在渲染函数`render`中需要执行多行代码的话，则需要`{}`进行包裹

#### 表格类输出 {#shiny_2_2_2}

1. tableOutput()

   输出静态表格，一次性展示完所有数据，常与渲染函数`renderTable()`搭配
   
2. dataTableOutput()

   输出动态表格，可进行翻页，常与渲染函数`renderDataTable()`搭配

> `dataTableOutput()`被弃用了，可以试试`DT::DTOutput()`

#### 图像类输出 {#shiny_2_2_3}

1. plotOutput()

   输出图像，常与渲染函数`renderPlot`搭配

> 建议设置`renderPlot(res = 96)`

#### 下载文件 {#shiny_2_2_4}

1. downloadButton()

2. downloadLink()

## 反应式编程 {#shiny_3}

反应式编程，简而言之，就是当输入变化时，所有相关的输出将会实时更新，而其余无关的输出则保持原状。

### 服务端 {#shiny_3_1}

回忆shiny应用程序的一般形式

```
library(shiny)

ui <- fluidPage(
  # front end interface
)

server <- function(input, output, session) {
  # back end logic
}

shinyApp(ui, server)
```

`ui`表示交互界面，呈现给每个用户的内容是相同的。`server`表示服务器端，由于每个用户输入的信息不尽相同，因此shiny程序在每次创建新会话(session)时都会独立地激活`server()`。

#### 输入 {#shiny_3_1_1}

对于ui中的输入`input`，会将其存储为类似列表的对象，每个元素的名称都是ui中相应的ID，在server中可通过`input$ID`来调用输入值。

> 注意`input`不能修改，只能读取

#### 输出 {#shiny_3_1_2}

同`input`，`output`也是类似列表的对象。对于你想输出的内容可通过`output$ID`的赋予其唯一标识符，然后在ui中的输出函数中根据ID进行调用。

> 注意`output$ID`赋值时一定要搭配渲染函数`render``

### 反应式编程语法 {#shiny_3_2}

