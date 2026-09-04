\ProvidesClass{template/v1}[2025/07/20 v1]
\LoadClass[12pt]{report}

% Core packages
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1,T5]{fontenc}
\RequirePackage{anyfontsize}
\RequirePackage{mathptmx}
\RequirePackage{enumitem}
\RequirePackage{xstring}

% Layout and headers
\RequirePackage{titlesec}
\RequirePackage[a4paper, top=15mm, bottom=15mm, left=25mm, right=20mm]{geometry}
\RequirePackage{indentfirst}
\RequirePackage{setspace}

% Graphics and tables
\RequirePackage{tabularx, makecell, multicol, multirow}
\RequirePackage{ragged2e, caption}
\RequirePackage{pgfgantt}
\RequirePackage{xcolor}

% Math packages
\RequirePackage{amsmath,amssymb}


\RequirePackage{iftex}

\RequirePackage{textcomp} % provide euro and other symbols
\RequirePackage{lipsum}
\RequirePackage{setspace}
\RequirePackage[table]{xcolor}
\RequirePackage{longtable,booktabs,array}
\RequirePackage{multirow}
\RequirePackage{calc} % for calculating minipage widths
% Correct order of tables after \paragraph or \subparagraph
\RequirePackage{etoolbox}

\RequirePackage{adjustbox}
\RequirePackage{graphicx}

\RequirePackage[
	colorlinks=true,
	pdfstartview=FitV,
	linkcolor=black,
	citecolor=blue,
	urlcolor=blue,
	breaklinks=true
]{hyperref}

\RequirePackage[maxcitenames=2, mincitenames=1, backend=biber,
sorting=none, bibencoding=utf8, natbib=true, maxbibnames=3, minbibnames=2]{biblatex}
% \DefineBibliographyStrings{english}{andothers={và các cộng sự}}
% \DefineBibliographyStrings{english}{and={và}}
\addbibresource{bibliography.bib}

%%%% GLOBAL STYLE SETTINGS %%%%%%
% Define custom colors
\definecolor{primaryblue}{RGB}{52, 152, 219}
\definecolor{secondarygreen}{RGB}{46, 204, 113}
\definecolor{warningorange}{RGB}{230, 126, 34}
\definecolor{criticalred}{RGB}{231, 76, 60}

\setlist[enumerate]{
    itemsep=4pt,
    parsep=8pt,
    topsep=0pt,
    partopsep=0pt,
    % Other options like leftmargin if you want to apply them globally
}

\setlength{\parskip}{1ex}
\renewcommand{\baselinestretch}{1.25}
\setcounter{secnumdepth}{-\maxdimen} % remove section numbering

\titlespacing{\section}{0in}{5pt}{3pt}
\titleformat{\section}
  {\large\bfseries}
  {\thesection}{0em}{}
\titleformat{\subsection}
  {\normalsize\bfseries}
  {\thesection}{1em}{}
\titleformat{\subsubsection}
  {\normalsize\bfseries}
  {\thesection}{1em}{}

\captionsetup[table]{name={Bảng}, labelsep=period, justification=raggedright, singlelinecheck=false, font=it}
\captionsetup[figure]{name={Hình},labelsep=period, font=it}

%%%% END GLOBAL STYLE SETTINGS %%%%%%

%%%%%%%%%% Title Module %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\Title}[2][local]{
    \expandafter\def\csname Title@#1\endcsname{#2}%
}
\newcommand{\@Title}[1][local]{
    \csname Title@#1\endcsname%
}
\newcommand{\@TitleUppercase}[1][local]{
    \expandafter\MakeUppercase\csname Title@#1\endcsname%
}
%%%%%%%%%% End Title Module %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Author module %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{AuthorCount}
\newcommand{\Author}[2]{%
    \stepcounter{AuthorCount}%
    \expandafter\gdef\csname authorname@\theAuthorCount\endcsname{#1}%
    \expandafter\gdef\csname authorid@\theAuthorCount\endcsname{#2}%
}

\newcommand{\@AuthorName}[1]{%
    \csname authorname@#1\endcsname%
}
\newcommand{\@AuthorId}[1]{%
    \csname authorid@#1\endcsname%
}
\newcommand{\@AuthorsStr}[1][, ]{%
    \setcounter{counter}{0}%
    \loop
        \stepcounter{counter}%
        \@AuthorName{\thecounter}\ifnum\thecounter<\theAuthorCountcount#1 \fi
        \ifnum\thecounter<\theAuthorCount
    \repeat
}
%%%%%%%%%% End Author module %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Advisor module %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{advisorcount}
\newcommand{\Advisor}[1]{%
    \stepcounter{advisorcount}%
    \expandafter\gdef\csname advisorname@\theadvisorcount\endcsname{#1}%
}
\newcommand{\@Advisor}[1]{%
    \csname advisorname@#1\endcsname%
}
\newcommand{\@AdvisorsStr}[1][, ]{%
    \setcounter{counter}{0}%
    \loop
        \stepcounter{counter}%
        \@Advisor{\thecounter}\ifnum\thecounter<\theadvisorcount#1 \fi
        \ifnum\thecounter<\theadvisorcount
    \repeat
}

\newcommand{\@AdvisorWithoutTitle}[1]{%
    \StrSubstitute{\csname advisorname@#1\endcsname}{PGS.}{}[\temp]%
    \expandafter\StrSubstitute{\temp}{GGS.}{}[\temp]%
    \expandafter\StrSubstitute{\temp}{ThS.}{}[\temp]%
    \StrSubstitute{\temp}{TS.}{}[\temp]%
    \expandafter\StrSubstitute{\temp}{KS.}{}[\temp]%
    \expandafter\StrSubstitute{\temp}{CN.}{}[\temp]%
    \temp
}

%%%%%%%%%% End Advisor module %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\Faculty#1{\gdef\@Faculty{#1}}
\def\ThesisFromDate#1{\gdef\@ThesisFromDate{#1}}
\def\ThesisToDate#1{\gdef\@ThesisToDate{#1}}

\newcounter{counter}%
%%%%%%% PRINT THE PROPOSAL INFORMATION %%%%%%%%%%%%
\newcommand{\printinfo}[1]{ %
    % National emblem and motto
    \begin{table}[ht]
    \hspace*{-0.5cm}
    \resizebox{\textwidth+1cm}{!}{%
        \begin{tabular}{cc}
            \begin{tabular}[c]{@{}c@{}}
                ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH \\
                TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN \\
                \textbf{\MakeUppercase{\@Faculty}} \\
                \raisebox{2ex}{\makebox[8cm]{\hrulefill}}
            \end{tabular} &
            \begin{tabular}[c]{@{}c@{}}
                \textbf{CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM} \\
                \textbf{Độc lập - Tự do - Hạnh phúc}  \\
                \raisebox{2ex}{\makebox[5cm]{\hrulefill}}
            \end{tabular}
        \end{tabular}
    }
    \end{table}
    
    % Main title with consistent font
    {\centering\fontsize{16pt}{15pt} \bfseries ĐỀ CƯƠNG CHI TIẾT\par}
    \vspace*{3ex}
    \hspace*{-1cm}
    \setlength{\extrarowheight}{2ex}
    \begin{tabular}{l p{11cm}}
        \textbf{TÊN ĐỀ TÀI TIẾNG VIỆT:} & {\@TitleUppercase} \\
        \textbf{TÊN ĐỀ TÀI TIẾNG ANH:} & {\@TitleUppercase[en]} \\
    \end{tabular}
    
    \vspace*{1ex}
    
    \noindent\textbf{Cán bộ hướng dẫn}: \@AdvisorsStr[, ]
    
    \noindent\textbf{Thời gian thực hiện: } Từ ngày \ifdefempty{\@ThesisFromDate}{.......................}{\@ThesisFromDate} đến ngày \ifdefempty{\@ThesisToDate}{.......................}{\@ThesisToDate}
    
    \noindent\normalfont\normalsize\textbf{Sinh viên thực hiện: }
    \ifnum \theAuthorCount = 1
        \normalfont\normalsize\textbf{\@AuthorId{1} - \@AuthorName{1}}
    \else
        % Print Authors
        \begin{enumerate}[leftmargin=2cm, itemindent=1.5em, itemsep=3pt, parsep=0pt]
        \setcounter{counter}{0}%
        \loop
            \stepcounter{counter}%
            \item \normalfont\normalsize\@AuthorId{\thecounter} - \@AuthorName{\thecounter}%
            \ifnum\thecounter<\theAuthorCount
        \repeat
        \end{enumerate}
    \fi
}

\newcommand{\printsignature}[0]{%
    \newcommand{\amp}{&}
    \newcounter{SignatureCount}
    \setcounter{SignatureCount}{\value{AuthorCount}}
    \addtocounter{SignatureCount}{\value{advisorcount}}

    \setlength{\extrarowheight}{0ex}
    \hspace*{-1cm}
    \begin{tabularx}{\textwidth+1cm}{*{\value{SignatureCount}}{>{\centering}X}}
    \setcounter{counter}{0}
    \whiledo {\value{counter}<\value{advisorcount}}{%
        \stepcounter{counter}
        \begin{tabular}{c}
          \textbf{Giảng viên hướng dẫn}\\
          \small\textit{(Ký và ghi rõ họ tên) }\\
          \vspace{3cm}\\
          \textbf{\@AdvisorWithoutTitle{\thecounter}}
        \end{tabular} \amp
    }

    \setcounter{counter}{0}
    \whiledo {\thecounter<\theAuthorCount}{%
        \stepcounter{counter}
        \begin{tabular}{c}
          \textbf{Sinh viên \thecounter} \\
          \footnotesize\textit{(Ký và ghi rõ họ tên) }\\
          \vspace{3cm} \\
          \textbf{\@AuthorName{\thecounter}}
        \end{tabular}
        \ifnum\thecounter<\theAuthorCount \noexpand\amp \fi
    }
    \end{tabularx}
}