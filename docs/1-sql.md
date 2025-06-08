# SQL {#sql}

该章节为《MySQL必知必会》的学习笔记。

## 连接MySQL {#sql_1}

### R {#sql_1_1}


``` r
library(DBI)
library(RMySQL)

con <- dbConnect(
  RMySQL::MySQL(),
  dbname = "data_1",   # 数据库名称（Navicat中的数据库名）
  host = "localhost",         # 服务器地址（本地为localhost，远程为IP）
  port = 3306,                # 端口号（默认3306）
  user = "root",     # 用户名（Navicat连接使用的）
  password = "password"  # 密码
)

dbListTables(con)    # 查看所有表

df <- dbReadTable(con, "table1")
```

用`dbReadTable()`读取整个表，并存储为`data frame`格式的数据框，后续可用`dplyr`进行数据清洗。

当然，可以直接使用`dbGetQuery()`进行SQL查询，第一个参数是连接名称，第二个参数就是SQL语句。


``` r
df <- dbGetQuery(con, "
  SELECT *
  FROM table1
  WHERE price > 100
")
```

亦或者，用`dplyr`语法代替SQL语句进行查询。


``` r
df <- tbl(con, 'table1')    # 懒加载，构建连接，数据还未导入R中
result_df <- df %>% 
  filter(price > 100) %>%   # dplyr语法筛选数据
  collect()                 # 将数据库中的数据导入到R中
```

> 这样导出的数据就是`tibble`格式的数据框，注意别忘了`collect()`

## MySQL必知必会 {#sql_2}

### 提要 {#sql_2_1}

- sql语句可单行或多行，以分号结尾

- 可使用空格缩进来增强可读性

- **MySQL**数据库的SQL语句**不区分大小写**，但关键字还是建议用大写

- 单行注释用`#`或`--`，多行注释用`/* text */`

- 子句顺序

   WHERE → GROUP BY → HAVING → SELECT（包括别名定义）→ ORDER BY → LIMIT

### 选择数据库与表 {#sql_2_2}

1. 选择数据库

```
USE 数据库名;
```

2. 查询所有可用的数据库

```
SHOW DATABASES;
```

3. 查询当前选择的数据库内可用表的列表

```
SHOW TABLES;
```

4. 查询某表的所有列

```
SHOW COLUMNS 
FROM 表名;
```

> 等价于`DESCRIBE 表名;`

### 检索数据 {#sql_2_3}

1. 检索单个列

```
SELECT 列名 
FROM 表名;
```

> 返回的数据顺序不一定与原始顺序一致

> select语句包含的内容为空，则输出`NULL`；若空内容出现在from语句，则select语句输出空值

2. 检索多个列

```
SELECT 列名_1, 列名_2, 列名_3 
FROM 表名;
```

3. 检索所有列

```
SELECT * 
FROM 表名;
```

4. 行去重

```
SELECT DISTINCT 列名 
FROM 表名;
```

> 无论检索多少列，`DISTINCT`作用于所有的列，而不单单是前置它的列。

5. 限制结果

`LIMIT`子句用于控制输出的行数，且记第一行为**行0**。

```
SELECT 列名 
FROM 表名 
LIMIT 5;
```

表示输出不多于5行的记录。

```
SELECT 列名 
FROM 表名 
LIMIT 5, 4;
```

表示从行5（事实上是第6行）开始数4行。

等价于

```
SELECT 列名 
FROM 表名 
LIMIT 4 OFFSET 5;
```

6. 使用完全限定的表名

可使用`X.Y`的形式来声明`Y`是来自于`X`的`Y`，其中`X`可为表名或数据库名

```
SELECT 表名.列名 
FROM 数据库名.表名;
```

7. 多分类表达式

```
SELECT
CASE
    WHEN condition THEN result
    ELSE
END AS col
```

### 排序检索数据 {#sql_2_4}

`ORDER BY`子句控制排序。

```
SELECT 列名 
FROM 表名
ORDER BY 列名;
```

若按多列排序，则为`ORDER BY 列名1, 列名2;`，先根据列名1排序，再根据列名2排序。

默认升序，若想降序则在列名后跟`DESC`关键字，并且`DESC`只作用于其所跟的列。

```
SELECT 列名 
FROM 表名
ORDER BY 列1 DESC, 列2;
```

有时需要对数据进行排名，下面介绍常见的三个窗口函数`ROW_NUMBER()``RANK()``DENSE_RANK()`。其中`PARTITION`表示分组标识，`ORDER BY`表示顺序标识。

`ROW_NUMBER()`直接给行编号，有多少行编多少号，如1、2、3、4。

`RANK()`对于相同值则相同排名，同时跳过排名，如1、1、3、4。

`DENSE_RANK()`对于相同值则相同排名，但不跳过排名，如1、1、2、3。

```
ROW_NUMBER() OVER ([PARTITION by col] ORDER BY col [ASC|DESC]) AS col
RANK() OVER ([PARTITION by col] ORDER BY col [ASC|DESC]) AS col
DENSE_RANK() OVER ([PARTITION by col] ORDER BY col [ASC|DESC]) AS col
```

> 返回的结果已经按ORDER BY进行排序

### 过滤数据 {#sql_2_5}

`WHERE`子句筛选符合条件的记录。

1. 常规操作符

```
SELECT 列名 
FROM 表名 
WHERE 列名 = 5;
```

除了相等是`=`，其余大小比较符号都符合常用习惯。还有范围操作符`BETWEEN X AND Y`，表示介于`X`和`Y`之间的数值，包括两端。

特别的，返回值为空值`NULL`的记录：`WHERE 列名 IS NULL;`。

2. `AND`或`OR`

同样，可使用`AND`或`OR`来组合多个条件。

```
SELECT 列名 
FROM 表名 
WHERE 列1 = 5 AND 列2 = 4;
```

> 默认优先处理`AND`关键字，必要时可用括号`()`将条件进行分组

3. `IN`

`IN`如同R里的`%in%`，筛选值是否在后续条件中，条件用圆括号包围。

```
SELECT 列名 
FROM 表名 
WHERE 列1 IN (2001, 2003);
```

4. `NOT`

`NOT`为取反操作，可对`IN`、`BETWEEN`、`EXISTS`等关键字取反。如`WHERE 列1 NOT IN (2001, 2003);`

5. `LIKE`

> `LIKE`并不是操作符，而是**谓词**，在技术上有所区别，但结果是相同的

顾名思义，`LIKE`查找与条件值像不像的记录。常与`%`和`_`搭配，前者表示匹配任意个字符，后者表示匹配单个字符。

```
SELECT 列名 
FROM 表名 
WHERE 列1 LIKE 'jet%';
```

> MySQL在配置方式中可设置是否区分大小写
> `LIKE`无法匹配`NULL`
> 一般不要把通配符放在pattern的开端，检索速度慢

6. 正则表达式

`REGEXP`关键字后跟正则表达式。

> `LIKE`是针对整个列值，而`REGEXP`则适合列值内，若`LIKE`不使用通配符的话，则二者存在差异
> 
> `REGXEP`可用`REGEXP BINARY`来实现区分大小写的功能，默认不区分大小写

   - `|`表示“或”
   
   - `[单个字符集]`与`[^单个字符集]`表示匹配其中一个字符或否定
   
   - `[1-9]`或`[a-z]`表示某个范围的单个字符集
   
   - 转义符`\\`
   
   - 事先预定的字符集，如`[:alnum:]`表示任意字母和数值
   
   - `.`、`*`、`+`、`?`、`{n}`、`{n,}`、`{n,m}`
   
   - `^`、`$`、`[[:<:]]`、`[[:>:]]`

> 可使用`SELECT '字符串' REGEXP 正则表达式`来简单检验正则表达式是否正确

### 计算字段{#sql_2_6}

1. 拼接字段

`Concat()`可用于列与列的拼接，类似`unite()`，并可用关键字`AS`设置别名，从而进行索引。

> 可用`Trim()`、`RTrim()`、`LTrim()`去掉空格

```
SELECT Concat(RTrim(列1), '_', LTrim(列2)) AS 列3
FROM 表名;
```

2. 计算字段

可在选择列时直接对列之间进行计算，如`mutate()`可直接在列之间进行计算一样，对新的列用`AS`命名即可。

```
SELECT 列1 * 列2 AS 列3
FROM 表名;
```

### 函数 {#sql_2_7}

不同SQL之间的SQL语句差异较小，而函数差异则较大。在使用函数时直接作用在列名上即可。

1. 文本处理函数

   - 字符串长度
   
      `LENGTH()`与`CHAR_LENGTH()`略有差异，前者返回字节长度，后者才是真的字符串长度。

2. 日期处理函数

   - DATEDIFF(date1,date2)

      返回date1-date2的天数

3. 数值处理函数

4. 窗口函数

窗口函数先进行分组及排序操作后再在窗口内执行对应的函数。

可以适当了解**窗口函数框架规范**。

> 窗口函数并不合并行，保留原始数据的完整性，例如在窗口中使用`COUNT`函数，则窗口内的每行赋值同一个数值

   - ROW_NUMBER()
   
      行编号
   
   - RANK()
   
      排名跳号
   
   - DENSE_RANK()

      排名不跳号
    
   - LAG(expression, offset, default_value)
   
      offest表示返回**前多少行**，默认为1；default_value表示若超出范围时的默认值，默认为NULL
      
   - LEAD(expression, offset, default_value)
   
      offest表示返回**后多少行**，默认为1；default_value表示若超出范围时的默认值，默认为NULL
      
   - FIRST_VALUE(expression)
   
      输出按指定顺序排列后的第一个值

> 注意，若想查询分组后的第一个值，必要时需在主键前添加DISTINCT
      
   - LAST_VALUE(expression)

5. 其他函数

   - IFNULL(expression, alt_value)
   
      NULL值替换，若为NULL值则替换为`alt_value`
      
   - IF(condition, value_if_true, value_if_false)

> IF函数可以SUM函数结合起来用于计数

### 汇总数据 {#sql_2_8}

下面提到的汇总函数也可作为窗口函数使用。

1. `AVG()`

   求某列均值，以列名为输入值，仅能输入一个列，并且忽略NULL。

> 能够处理布尔表达式，如`AVG(c.action = 'confirmed')`

2. `COUNT()`

   对行数进行计数，若为`COUNT(*)`则无论是否有NULL都算进去，若为`COUNT(列)`则忽视NULL值。
   
3. `MAX()`

4. `MIN()`

5. `SUM()`

   可根据算术符对多个列进行求和，如`SUM(列1 * 列2)`，忽略NULL。
   
> 有时可与`DISTINCE`关键字结合起来，如`AVG(DISTINCT 列1)`

### 数据分组 {#sql_2_9}

`GROUP BY`子句进行分组，可同时对多列进行分组，并在最后规定的分组上汇总。

在使用`GROUP BY`时，`SELECT`中的每个列（除了聚集计算语句外）都得在`GROUP BY`中给出（若是表达式则不能使用别名，而是指定相同的表达式）。

> 否则会按列的默认顺序返回对应的函数，从而导致“记录”的值不匹配

```
SELECT 列1, COUNT(*) AS 列
FROM 表名
GROUP BY 列1;
```

> `GROUP BY 列 WITH ROLLUP`将会根据分组顺序逐级向上汇总，在返回的数据中增加汇总行

若要对分组进行筛选，可使用`HAVING`子句。与`WHERE`子句的区别在于，`WHERE`是对行进行过滤，而`HAVING`可以过滤分组。

```
SELECT 列1, COUNT(*) AS 列
FROM 表名
WHERE 列2 >= 10
GROUP BY 列1
HAVING COUNT(*) >= 2;
```

`ORDER BY`也能对分组后的数据进行排序。

### 子查询 {#sql_2_10}

所谓**子查询**，也就是`SELECT`语句的嵌套（用圆括号包围），将内层`SELECT`结构的结果作为条件输出给外层`SELECT`结构进行查询。

1. 过滤

子查询作为条件传给外层的`WHERE`子句。

```
SELECT 列1
FROM 表1
WHERE 列2 IN (SELECT 列3
              FROM 表2
              WHERE 列4 = 'xxx');
```

> 除了`IN`，还可以与其他操作符结合在一起

2. 计算字段

除了作为过滤条件外，还可用于计算字段传给外层的`SELECT`子句。

```
SELECT 列1,
       (SELECT COUNT(*)
        FROM 表2
        WHERE 表2.列2 = 表1.列3) AS 列4
FROM 表1
ORDER BY 列1;
```

> 注意对于同名列需要区分表来源，这种情形也称之为**相关子查询**

### 表联结 {#sql_2_11}

**主键**，即某表中每条记录的唯一标识符，如ID。**外键**，即某表包含另一个表的主键的那一列，该列定义了两个表之间的关系。

为了简洁，可在`FROM`子句中用`AS`关键字为表取别名。

也可多次使用联结，并且每次联结的类型可以不同。

> 联结时除了匹配主键或外键，还可以用其他条件来联结，如`ON a.player_id = t.player_id AND DATEDIFF(a.event_date, t.first_date) = 1`

1. `WHERE`联结

在`WHERE`子句中添加两个表之间的配对关系（主键与外键），从而实现表联结。且该联结方式为**内部联结**。

```
SELECT 列1, 列2
FROM 表1, 表2
WHERE 表1.主键 = 表2.外键
ORDER BY 列1, 列2;
```

> 若想得到两个表之间的笛卡尔积（也就是**全联结**），则去掉`WHERE`子句即可

2. 内部联结

内部联结`INNER JOIN`取表的交集。

```
SELECT 列1, 列2, 列3
FROM 
表1 
INNER JOIN 表2 ON 表1.主键 = 表2.外键
INNER JOIN 表3 ON 表1.外键 = 表3.主键;
```

> 建议用`INNER JOIN`替代`WHERE`联结

3. 自联结

在自联结中，需要使用表别名。自联结的多张表中其中一张表作为主表，用于展示查询结果，其余表作为副表，用于提供查询条件。

```
SELECT t1.列1, t1.列2
FROM 表1 AS t1, 表1 AS t2
WHERE t1.列3 = t2.列3
AND t2.列4 = 'xxx';
```

> 这里通过列3来将t1和t2进行匹配，并通过t2.列4进行筛选，从而筛选出t1中符合条件的列

4. 自然联结

自然联结排除列的多次出现，使每个列只返回一次。因此，只需要选中主表的所有列（`SELECT 主表.*`），其余副表中的列明确指名即可。

5. 外部联结

外部联结包括左联结`LEFT OUTER JOIN`和右联结`RIGHT OUTER JOIN`，二者本质上是互通的。

> 左联结就是以`LEFT OUTER JOIN`左边的表为准，右联结则以`RIGHT OUTER JOIN`右边的表为准

```
SELECT 表1.列1, 表2.列2
FROM 表1 LEFT OUTER JOIN 表2
ON 表1.列1 = 表2.列1
```

### 组合查询 {#sql_2_12}

组合查询，就是将多条`SELECT`语句的结果整合到一起，作为单个查询结果返回。用关键字`UNION`连接多个`SELECT`语句。

> 组合查询时每个查询必须包含相同的列、表达式或聚集函数

组合查询即可用于对一张表的多次查询，也可用于在单次查询中从不同表返回类似结构的数据。

> 前一种情形可用多个`WHERE`子句实现相同效果

```
SELECT 列1, 列2
FROM 表1
WHERE 列2 <= 5
UNION
SELECT 列1, 列2
FROM 表1
WHERE 列3 IN (xxx,yyy);
```

在多次查询中，对与重复的行`UNION`会默认省去，使用`UNION ALL`关键字可返回所有匹配的行。

在对组合查询结果排序时只能在最后一条`SELECT`语句后面添加`ORDER BY`子句

### 全文本搜索 {#sql_2_13}

> 并非所有引擎都支持全文本搜索，MyISAM引擎支持全文本搜索，而InnoDB引擎则不支持

要实现全文本搜索，需要在创建表的时候将某些列指定为`FULLTEXT`，这样就能在后续使用`Match()`与`Against()`函数来进行搜索。`Match()`指定被搜索的列，该列必须在指定的`FULLTEXT`列中，若指定多列，则必须列出它们并且顺序与`FULLTEXT`中的一致。`Against()`指定要搜索的表达式，若要区分大小写则使用`BINARY`关键字。

> 不要在导入数据时使用`FULLTEXT`，而是等导入所有数据后再修改表，定义`FULLTEXT`

```
SELECT 列1
FROM 表名
WHERE Match(列1) Against('xxx');
```

> 被搜索的表达式越早出现，则其在搜索结果中越靠前

- 查询扩展

   使用全文本搜索时只能找到包含目标表达式的行，而使用查询扩展则会在全文本搜索的基础上找出与目标较为相似（即使没有目标表达式，但出现其他共同的词汇）的其他行。
   
```
SELECT 列1
FROM 表名
WHERE Match(列1) Against('xxx' WITH QUERY EXPANSION);
```

- 布尔文本搜索

   能更为精细地控制搜索模式，请详见原文。

### 插入数据 {#sql_2_14}

1. 插入记录

使用`INSERT`子句插入记录。内容可填充为`NULL`，

```
INSERT INTO 表名 (
列名
)
VALUES(
NULL,
'xxx',
...
);
```

在填充时如果不给出列名（也就是表名后的圆括号），则按列的默认顺序依次填充。

> 对于被定义为主键的列，可以不用为其填充，MySQL会自动填充

若要插入多行记录，则确保值的顺序与列的顺序一致，多行记录的值用圆括号括起来并用逗号隔开。

```
INSERT INTO 表名 (
列名
)
VALUES(
NULL,
'xxx',
...
),
(
NULL,
'xxx',
...
);
```

2. 插入检索出的数据

可以先用`SELECT`检索出数据，再把检索出的数据插入到另一个表中。

```
INSERT INTO 表1 (
...
)
SELECT ...
FROM 表2;
```

注意`SELECT`子句中的列名不一定要与表1中的列名一样，在执行`INSERT SELECT`结构时仅关心`SELECT`子句中的位置关系，根据对应位置进行填充。

> 如果其中包含了主键列，你可以省略该列，MySQL会自动增量
> 
> `SELECT`子句也可包含`WHERE`子句

### 更新数据 {#sql_2_15}

1. 更新

`UPDATE`语句用于更新数据。更新规则为“列 = 值”的形式，不同列之间用逗号隔开，利用`WHERE`筛选出要更新的行。

```
UPDATE 表名
SET 列1 = 'xxx',
    列2 = 'yyy'
WHERE 列3 = 'zzz';
```

> 在更新过程中可能会报错，默认会恢复到原值，如果想无视错误而更新一部分的话，可使用`UPDATE IGNORE`语句
> 
> 如果想删除某个列的值，可设置它为`NULL`

2. 删除

`DELETE`语句删除行。

```
DELETE FROM 表名
WHERE 列 = '...';
```

根据`WHERE`子句删除特定行，若没有`WHERE`则删除所有行，但不意味着删除该表。

### 表的操作 {#sql_2_16}

1. 创建表

利用`CREATE TABLE`创建表。

```
CREATE TABLE 表名 IF NOT EXISTS
(
主键1 int NOT NULL AUTO_INCREMENT,
主键2 int NOT NULL,
列1 char(50) NOT NULL DEFAULT 'xxx',
列2 char (5) NULL,
PRIMARY KEY (主键1, 主键2)
) ENGINE = InnoDB;
```

创建表时，需给出列名及其对应的数据类型。可根据`PRIMARY KEY`来指定主键，主键可设置自动增量`AUTO_INCREMENT`。

> 每个表只有一列允许`AUTO_INCREMENT`
> 
> 可根据`SELECT last_insert_id()`来返回最后一个`AUTO_INCREMENT`值
> 
> 可用`DEFAULT`来设置默认值
> 
> InnoDB引擎支持事务处理，但不支持全文本搜索；MEMORY速度快，适合临时表；MyISAM支持全文本搜索，但不支持事务处理

2. 更新表

> 当表中存储数据以后，不建议再更新，因此在创建表时就得深思熟虑

使用`ALTER TABLE`语句进行更改表操作。对单个表进行多次更改时，可用逗号隔开每次更改。

```
ALTER TABLE 表名
ADD 列名 CHAR(20);

ALTER TABLE 表名
DROP COLUMN 列名;

ALTER TABLE 表1
ADD CONSTRAINT 外键约束名
FOREIGN KEY (外键字段名)
REFERENCES 表2 (表2主键);
```

> 外键字段名是表中已经存在的列名，外键约束名则是该外键在整个数据库中的唯一标识符

3. 删除表

```
DROP TABLE 表名;
```

4. 重命名表

```
RENAME TABLE 表1 TO 新表1,
             表2 TO 新表2;
```

### 使用视图 {#sql_2_17}

视图(view)就是一张虚拟表，既可又一张表构成，又可结合多个表的数据。

1. 创建视图

```
CREATE VIEW 视图名 AS
SELECT 列1, 列2, 列3
FROM 表1, 表2, 表3
WHERE 表1.主键 = 表2.表1外键
AND 表3.主键 = 表2.表3外键
```

> `AS`前面的就是视图名

在创建好视图后，即可对视图进行一般的查询操作。

> 一般视图用于检索数据而非更新，故不介绍视图的更新操作

### 使用存储过程 {#sql_2_18}

> **存储过程**类似自定义函数，你不一定有创建存储过程的权限，但可能会使用到它
> 
> **游标**是包含在存储过程中的一个概念，故不介绍

使用`CALL`关键字来调用存储过程。

```
CALL 存储过程名
(@参数1, @参数2, @参数3);
```

### 触发器 {#sql_2_19}

> 创建触发器也是需要安全访问权限，故了解即可

所谓触发器，就是在特定语句（仅限`DELETE`、`INSERT`、`UPDATE`）发生之前或之后自动执行的行为。

1. 创建触发器

```
CREATE TRIGGER 触发器名 AFTER INSERT ON 表名
FOR EACH ROW 操作;
```

`CREATE TRIGGER`关键字来创建触发器，`AFTER INSERT`定义触发器在`INSERT`语句发生后触发，第二行就是针对每个被插入行的具体操作了，具体操作可到时视情况而定。

2. 删除触发器

```
DROP TRIGGER 触发器名;
```

3. 使用触发器

- INSERT触发器

   在INSERT触发器内部可引用一个名为`NEW`的虚拟表，用于访问被插入的行。

```
CREATE TRIGGER 触发器名 AFTER INSERT ON 表名
FOR EACH ROW SELECT NEW.列1;
```

该代码表明在每次插入新的行后，返回这些行的列1。

- DELETE触发器

   在DELETE触发器内部可引用一个名为`OLD`的虚拟表，用于访问要被删除的行。

```
CREATE TRIGGER 触发器名 BEFORE DELETE ON 表名
FOR EACH ROW
BEGIN
   ...
END;
```

该代码定义了在`DELETE`语句前要执行的行为，具体行为被包含在`BEGIN END`块中，这使得能够该触发器能够容纳多条SQL语句。

- UPDATE触发器

   在UPDATE触发器内部可引用`OLD`和`NEW`的虚拟表，用于访问被插入与被删除的行。

```
CREATE TRIGGER 触发器名 BEFORE UPDATE ON 表名
FOR EACH ROW 操作;
```

### 事务处理 {#sql_2_20}

事务处理一种机制，确保某些操作应当成批执行，或者不执行，从而确保结果的完整性与过程的一致性。

> **事务**指的是一组SQL语句
> 
> 事务处理用来管理`INSERT`、`UPDATE`、`DELETE`语句

事务处理以`START TRANSACTION`关键字开头。

- 回退

`ROLLBACK`关键词撤销前面的所有操作，表的状态会回到`START TRANSACTION`前的状态。

```
START TRANSACTION;
操作;
ROLLBACK;
```

- 提交

`COMMIT`是在前述所有操作成功的前提下才会执行，得到最终结果。

```
START TRANSACTION;
操作;
COMMIT;
```

- 保留点

保留点相当于存档点，有时候不一定要回退到最开始的状态，可以中途定义一些保留点，然后回退到保留点即可。

```
START TRANSACTION;
操作;
SAVEPOINT 保留点1;
操作;
ROLLBACK TO 保留点1;
```
