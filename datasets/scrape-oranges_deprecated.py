# import urllib
# resource = urllib.urlopen("http://www.digimouth.com/news/media/2011/09/google-logo.jpg")
# output = open("file01.jpg","wb")
# output.write(resource.read())
# output.close()
#
#
# from bs4 import BeautifulSoup, SoupStrainer
#
#
# with open('orange_urls.txt','r') as f:
#     for img in BeautifulSoup(f.read(), parse_only=SoupStrainer('a')):
#         # if link.has_attr('href')
#         print(link)


import urllib.request
with open('orange_urls.txt') as f:
    content = f.readlines()
    # print(content)
# you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]
i = 0;
save_filename = "orange";
for row in content:
    try:
        save_filename = "orange_";
        with urllib.request.urlopen(row) as url:
            i = i+1;
            stream = url.read()
            save_filename +=str(i);
            save_file = open(save_filename,'wb')
            save_file.write(stream)
            save_file.close()
    except:
        print("something broke, yo")
    # print(s)


# import urllib
# url = "http://farm1.static.flickr.com/40/81014064_ff7a1ca6c0.jpg"
# uopen = urllib.urlopen(url)
#
# stream = uopen.read()
# file = open('filename','w')
# file.write(stream)
# file.close()
