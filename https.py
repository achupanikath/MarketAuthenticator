import urllib.parse
from OpenSSL import crypto
from collections import namedtuple
from http.client import HTTPResponse
import cgi
import socket
import ssl
import urllib
import certifi
import cryptography
from cryptography import x509

CA_CERTS = certifi.where()


def sslcert(host, port):
    # from https://www.programcreek.com/python/example/62606/ssl.get_server_certificate
    try:
        cert = ssl.get_server_certificate((host, port))
        print(cert)
        x509 = crypto.load_certificate(crypto.FILETYPE_PEM, cert)
        issuer = x509.get_issuer()
        print(issuer)
        print(x509)
        dump = crypto.dump_certificate(crypto.FILETYPE_PEM, x509)
        print(dump)
        cert_hostname = x509.get_subject().CN
        print(cert_hostname)
    except:
        print("Error")


def check_host_name(peercert, split):
    # code adapted from
    # https://docs.fedoraproject.org/en-US/Fedora_Security_Team/1/html/Defensive_Coding/sect-Defensive_Coding-TLS-Client-Python.html

    """Simple certificate/host name checker.  Returns True if the
    certificate matches, False otherwise.  Does not support
    wildcards."""
    # Check that the peer has supplied a certificate.
    # None/{} is not acceptable.

    names = []
    split = format_url(url)
    name = split[0][1]
    names.append(name)
    if (name[0:4] == "www."):
        names.append(name[4:])
    if not peercert:
        return False

    if (not isinstance(peercert, crypto.X509)):
        if (not isinstance(peercert, str)):
            peercert = ssl.DER_cert_to_PEM_cert(peercert)
            peercert = bytes(peercert, encoding='utf-8')
        peercert = crypto.x509.load_pem_x509_certificate(peercert)

    subjalt_ext = cryptography.x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
    ext = peercert.extensions.get_extension_for_oid(subjalt_ext)
    # Get the dNSName entries from the SAN extension
    dns = ext.value.get_values_for_type(cryptography.x509.DNSName)
    if ((x in dns for x in names) or (x == dns for x in names)):
        return True
    else:
        # Only check the subject DN if there is no subject alternative
        # name.
        subj = peercert.subject
        cn = subj.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
        if (cn in names):
            return True
    return False


def test(cert):
    peercert = ssl.DER_cert_to_PEM_cert(cert)
    peercert = bytes(peercert, encoding='utf-8')
    cert = crypto.x509.load_pem_x509_certificate(peercert)
    print(cert.not_valid_before)
    print(cert.not_valid_after)
    print(cert.issuer)
    print(cert.subject)
    print(hex(cert.serial_number))


def get_certificate(url):
    # code adapted from
    # https://chaobin.github.io/2015/07/22/a-working-understanding-on-SSL-and-HTTPS-using-python/

    results = format_url(url)
    for https, addr, path in results:
        port = 443 if https else 80
        cert = ssl.get_server_certificate((addr, port))
        # cert = bytes(cert, encoding= 'utf-8')
        if cert is not None:
            return cert
        '''
        cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert)
        issuer = cert.get_issuer()
        print(issuer)
        subject = cert.get_subject()
        print(subject)
        components = dict(subject.get_components()) # convert to dict
        print(components)'''


def convert_certifi_to_PEM_list(trusted_certs=CA_CERTS):
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=trusted_certs)
    _certs = context.get_ca_certs(binary_form=True)
    certs = []
    for cert in _certs:
        certs.append(ssl.DER_cert_to_PEM_cert(cert))
    return certs


def verify_certificate_chain(cert, trusted_certs=None):
    # code adapted from
    # https://aviadas.com/blog/2015/06/18/verifying-x509-certificate-chain-of-trust-in-python/

    # Create a certificate store and add your trusted certs
    try:
        # Download the certificate from the url and load the certificate
        if trusted_certs is None:
            trusted_certs = convert_certifi_to_PEM_list()

        store = crypto.X509Store()

        # Assuming the certificates are in PEM format in a trusted_certs list
        for _cert in trusted_certs:
            x509 = crypto.load_certificate(type=crypto.FILETYPE_PEM, buffer=_cert)
            store.add_cert(x509)

        # Create a certificate context using the store and the downloaded certificate
        if (not isinstance(cert, str)):
            cert = ssl.DER_cert_to_PEM_cert(cert)
        cert = crypto.load_certificate(type=crypto.FILETYPE_PEM, buffer=cert)
        store_ctx = crypto.X509StoreContext(store, cert)

        # Verify the certificate, returns None if it can validate the certificate
        store_ctx.verify_certificate()

        return True

    except Exception as e:
        print(e)
        return False


def handshake(sock):
    # code adapted from
    # https://chaobin.github.io/2015/07/22/a-working-understanding-on-SSL-and-HTTPS-using-python/
    new_sock = ssl.wrap_socket(sock,
                               ciphers="HIGH:-aNULL:-eNULL:-PSK:RC4-SHA:RC4-MD5",
                               ssl_version=ssl.PROTOCOL_TLS,
                               cert_reqs=ssl.CERT_REQUIRED,  # I will verify your certificate
                               ca_certs=CA_CERTS  # using a list of trusted CA's certificates
                               )
    return new_sock


def make_header(path, addr, encoding, CRLF):
    # code adapted from
    # https://chaobin.github.io/2015/07/22/a-working-understanding-on-SSL-and-HTTPS-using-python/

    headers = [
        'GET %s HTTP/1.1' % (path),
        'Host: %s' % addr,
        'User-Agent: Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
        'Charset: %s' % encoding
    ]
    header = '\n'.join(headers)
    header += CRLF
    return bytes(header, encoding=encoding)


def parse_response(resp, encoding='utf-8'):
    # code adapted from
    # https://chaobin.github.io/2015/07/22/a-working-understanding-on-SSL-and-HTTPS-using-python/

    Page = namedtuple('Page', ('code', 'headers', 'body'))
    resp.begin()
    code = resp.getcode()
    # assume status 200
    headers = dict(resp.getheaders())

    def get_charset():
        ctt_type = headers.get('Content-Type')
        return cgi.parse_header(ctt_type)[1].get('charset', encoding)

    body = resp.read()
    body = str(body, encoding=get_charset())
    return Page(code, headers, body)


def request(sock, path, addr, encoding='utf-8', CRLF='\r\n\r\n'):
    # code adapted from
    # https://chaobin.github.io/2015/07/22/a-working-understanding-on-SSL-and-HTTPS-using-python/

    try:
        header = make_header(path, addr, encoding, CRLF)
        sock.sendall(header)
        # receive it
        resp = HTTPResponse(sock)
        return parse_response(resp, encoding=encoding)
    except:
        # https failed
        print("ERROR: Request to {} FAILED".format(url))
        return None


def close_socket(sock):
    sock.shutdown(1)
    sock.close()


def get_socket_and_cert(url, port=None, get_cert_format=None, https=True):
    # code adapted from
    # https://chaobin.github.io/2015/07/22/a-working-understanding-on-SSL-and-HTTPS-using-python/

    # returns (socket, certificate, https)

    if https and port is None:
        port = 443
    if not https and port is None:
        port = 80

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    certificate = None
    try:
        sock.connect((url, port))
        if (https):
            sock = handshake(sock)
            if get_cert_format is not None:
                if (not isinstance(get_cert_format, bool)):
                    get_cert_format = True
                # if get_cert is False this will be a dict, else a PEM cert
                certificate = sock.getpeercert(binary_form=get_cert_format)
                return sock, certificate, True
        return sock, None, False
    except Exception as e:
        print('{} connection failed for {} because of:'.format("https" if https else "http", url))
        print("     " + str(e) + "\n")
        return None, None, None


def format_url(url):
    # formats url to be <https:// or http://><www.hostname><path>
    # and returns each of these components individually
    # if input url does not start with https:// or http://, function returns 2 urls, one for each
    components = urllib.parse.urlsplit(url)
    if (components.scheme == 'https'):
        HTTPS = True
    elif (components.scheme == 'http'):
        HTTPS = False
    else:
        HTTPS = None
        # make 2 urls, one for http and one for https and try both
        # return format_url("https://" + url).extend(format_url("http://" + url))
        _, addr, path = format_url("https://" + url)[0]
        return [(True, addr, path), (False, addr, path)]

    if (components.path is not (None or '' or "") and components.path != url):
        path = '%s' % components.path
    else:
        path = '/'
    addr = components.hostname
    if (addr is None):
        addr = url
    elif (addr[0:4] != 'www.'):
        addr = "www." + addr

    return [(HTTPS, addr, path)]


def get_any_socket(url, get_cert_format=None):
    # tries to open a socket through https(preferred) or http
    # returns (socket, cert, https)
    urls = format_url(url)
    for HTTPS, addr, path in urls:
        results = get_socket_and_cert(addr, https=HTTPS, get_cert_format=get_cert_format)
        if results[0] is not None:
            # if valid socket
            return results
    print("Error, unable to connect to {} through http or https".format(url))
    return None, None, None


def get_any_socket_cert(url, get_cert_format=None):
    socket, cert, https = get_any_socket(url, get_cert_format)
    if cert is None and get_cert_format is not None:
        cert = get_certificate(url)
    return socket, cert, https


def check_all(url):
    sock = get_any_socket_cert(url, get_cert_format=True)
    print("Checking {}\n".format(url))
    print("Website supports https:          {}".format(str(sock[2])))
    print("Certificate retrieved:           {}".format(str(sock[1] is not None)))
    print("Certificate chain verified:      {}".format(str(verify_certificate_chain(sock[1]))))
    print("Hostname matches certificate:    {}".format(str(check_host_name(sock[1], url))))


# url = "https://aviadas.com/blog/2015/06/18/verifying-x509-certificate-chain-of-trust-in-python/"
# url = "https://expired.badssl.com/"
# url = "https://docs.python.org/3/tutorial/errors.html"
# url = "https://addidas.com/us"
url = "https://www.jetro.go.jp/"
#url = "https://www.nike.com"
#url = "nike.com"
#url = "www.nike.com/us"
#url = "expired.badssl.com/"

check_all(url)
'''
Abilities so far:
    make https socket
    make http socket
    request a webpage using http or https
    get certificate
    verify certificate chain
    verify hostname

attributes:
    https presence
    valid certificate chain
    valid hostname
    whether there is a certificate
'''