from bs4 import BeautifulSoup

def remove_span_tags(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # Remove all <span> tags
    for span_tag in soup.find_all('span'):
        span_tag.unwrap()

    return str(soup)

def main():
    # Example usage:
    html_string = '<p>This is a <span>sample</span> HTML <span>string</span>.</p>'
    result_string = remove_span_tags(html_string)
    print(result_string)

if __name__ == "__main__":
    main()

