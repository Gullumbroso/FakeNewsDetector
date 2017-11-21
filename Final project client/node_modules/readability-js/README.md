# Readability

Nodejs module for extracting web page content using Cheerio.

Turn any web page into a clean view. This module is based on [luin](https://github.com/luin/readability)'s readability project.

[![Build Status](https://travis-ci.org/Ournet/readability-js.png?branch=master)](https://travis-ci.org/Ournet/readability-js)

## Install

```
npm install readability-js
```

## Usage

```
read(html [, options], callback)
```

**Where**

- **html** url or html code.
- **options** is an optional options object
- **callback** is the callback to run - callback(error, article, meta)

## Example

```
var read = require('readability-js');

read('http://howtonode.org/really-simple-file-uploads', function(err, article, meta) {
  // Main Article
  console.log(article.content.text());

  // Title
  console.log(article.title);

  // Article HTML Source Code
  console.log(article.content.html());
});
```

**NB** If the page has been marked with charset other than utf-8, it will be converted automatically. Charsets such as GBK, GB2312 is also supported.

## Options

readability-js will pass the options to request directly. See request lib to view all available options.

readability-js has 2 additional options:

- **onlyArticleBody** (Boolean) - get only article body or all main content;

- **preprocess** - which should be a function to check or modify downloaded source before passing it to readability.

```
read(url, {
  preprocess: function(source, response, contentType, callback) {
    if (source.length > maxBodySize) {
      return callback(new Error('too big'));
    }
    callback(null, source);
  }, function(err, article, response) {
    //...
  });
```

### Article object

- **content** - The article content of the web page. Return false if failed. Is a Cheerio object.

- **title** - The article title of the web page. It's may not same to the text in the `<title>` tag.

- **excerpt** - The article description from any description, og:description or twitter:description `<meta>`
