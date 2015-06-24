package edu.zju.dd.crawler;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.http.Header;
import org.apache.http.HttpEntity;
import org.apache.http.HttpException;
import org.apache.http.HttpHeaders;
import org.apache.http.HttpHost;
import org.apache.http.HttpRequest;
import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.ProtocolException;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.Credentials;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.AuthCache;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.CookieStore;
import org.apache.http.client.CredentialsProvider;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpUriRequest;
import org.apache.http.client.params.ClientPNames;
import org.apache.http.client.params.CookiePolicy;
import org.apache.http.client.protocol.ClientContext;
import org.apache.http.client.protocol.HttpClientContext;
import org.apache.http.conn.ClientConnectionManager;
import org.apache.http.impl.auth.BasicScheme;
import org.apache.http.impl.client.BasicAuthCache;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.impl.client.DefaultRedirectStrategy;
import org.apache.http.params.HttpParams;
import org.apache.http.protocol.BasicHttpContext;
import org.apache.http.protocol.ExecutionContext;
import org.apache.http.protocol.HttpContext;
import org.apache.http.util.EntityUtils;

import org.apache.http.Consts;
import org.apache.http.HttpEntity;
import org.apache.http.NameValuePair;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.cookie.Cookie;
import org.apache.http.impl.client.BasicCookieStore;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.util.EntityUtils;

import edu.zju.dd.codeforce.meta.Status;
import edu.zju.dd.codeforce.meta.Submission;
import edu.zju.dd.codeforce.statusparser.StatusParser;
import edu.zju.dd.codeforce.statusparser.SubmissionParser;


public class HDOJSourceCrawler {

	//httpClient ģ��?�ύ
	public static void diandianAdd() throws Exception{
		DefaultHttpClient httpClient = new DefaultHttpClient();
		String url = "http://acm.hdu.edu.cn//vip/userloginex.php?action=login&cid=479";
		HttpPost post = new HttpPost(url);
		//post.addHeader("username", "admin");
		//post.addHeader("userpass", "921@njust");
		
		httpClient.getParams().setParameter("username", "admin");
		httpClient.getParams().setParameter("userpass", "921@njust");
		
		
		HttpResponse response = null;
		try {
			response = httpClient.execute(post);
		} catch (IOException e) {
			e.printStackTrace();
		}
	
      
	
		post.abort();
		
		
		
		
		//HttpGet get = new HttpGet("http://acm.hdu.edu.cn/vip/viewcode.php?rid=8506&cid=479");
		HttpGet get = new HttpGet("http://acm.hdu.edu.cn/vip/contest_list.php");
		HttpResponse response2 = httpClient.execute(get);
		System.out.println(response2.toString());
		
		Header headers[] = response2.getAllHeaders();
		for(int i=0;i<headers.length;i++) {
			System.out.println(headers[i].toString());
		}
		System.out.println("----------------------------------------------");
		
		HttpEntity entity = response2.getEntity();
		System.out.println(EntityUtils.toString(entity));
	}
	
	
	public static void testHttpGet(String url) throws Exception{
		HttpClient httpClient = new DefaultHttpClient();
		HttpGet httpGet = new HttpGet(url);
		HttpResponse response = httpClient.execute(httpGet);
		
		System.out.println("statusLine = "+response.getStatusLine());
		
		System.out.println("headers begin-----------------------------------");
		Header header[] = response.getAllHeaders();
		for(int i=0;i<header.length;i++) {
			System.out.println(header[i]);
		}
		System.out.println("headers end-------------------------------------");
		HttpEntity entity = response.getEntity();
		System.out.println(EntityUtils.toString(entity, "gbk"));
	}
	
	public static String getHtml(String url) throws Exception{
		HttpClient httpClient = new DefaultHttpClient();
		HttpGet httpGet = new HttpGet(url);
		HttpResponse response = httpClient.execute(httpGet);
		HttpEntity entity = response.getEntity();
		return EntityUtils.toString(entity, "gbk");
	} 
	
	public static void check() throws Exception{
		CloseableHttpClient httpclient = new CloseableHttpClient() {
			
			@Override
			public void close() throws IOException {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public HttpParams getParams() {
				// TODO Auto-generated method stub
				return null;
			}
			
			@Override
			public ClientConnectionManager getConnectionManager() {
				// TODO Auto-generated method stub
				return null;
			}
			
			@Override
			protected CloseableHttpResponse doExecute(HttpHost arg0, HttpRequest arg1,
					HttpContext arg2) throws IOException, ClientProtocolException {
				// TODO Auto-generated method stub
				return null;
			}
		};

		HttpHost targetHost = new HttpHost("acm.hdu.edu.cn", 80, "http");
		CredentialsProvider credsProvider = new BasicCredentialsProvider();
		credsProvider.setCredentials(
		        new AuthScope(targetHost.getHostName(), targetHost.getPort()),
		        new UsernamePasswordCredentials("admin", "921@njust"));

		// Create AuthCache instance
		AuthCache authCache = new BasicAuthCache();
		// Generate BASIC scheme object and add it to the local auth cache
		BasicScheme basicAuth = new BasicScheme();
		authCache.put(targetHost, basicAuth);

		// Add AuthCache to the execution context
		HttpClientContext context = HttpClientContext.create();
		context.setCredentialsProvider(credsProvider);

		HttpPost httpPost = new HttpPost("/vip/userloginex.php?action=login&cid=479");
		for (int i = 0; i < 3; i++) {
		    CloseableHttpResponse response = httpclient.execute(
		            targetHost, httpPost, context);
		    try {
		        HttpEntity entity = response.getEntity();

		    } finally {
		        response.close();
		    }
		}
	}
	
	@SuppressWarnings("deprecation")
	public static void test() throws Exception {
        BasicCookieStore cookieStore = new BasicCookieStore();
        CloseableHttpClient httpclient = HttpClients.custom().setDefaultCookieStore(cookieStore).build();
        /*
        ttpGet httpGet; 
        HttpResponse response;
        
        httpGet = new HttpGet("http://acm.hdu.edu.cn/vip/contest_list.php");
        response = httpclient.execute(httpGet);
        EntityUtils.consume(response.getEntity());
        
        httpGet = new HttpGet("http://acm.hdu.edu.cn/vip/2013nanjing/");
        response = httpclient.execute(httpGet);
        EntityUtils.consume(response.getEntity());
        
        httpGet = new HttpGet("http://acm.hdu.edu.cn/vip/userloginex.php?cid=479");
        response = httpclient.execute(httpGet);
        EntityUtils.consume(response.getEntity());
        
        */
        
        try {
            
            HttpGet httpget = new HttpGet("http://acm.hdu.edu.cn/vip/userloginex.php?cid=479");
        	//HttpGet httpget = new HttpGet("http://www.rfp.ca/login/");
        	
            CloseableHttpResponse response1 = httpclient.execute(httpget);
            try {
                HttpEntity entity = response1.getEntity();

                System.out.println("Login form get: " + response1.getStatusLine());
               // System.out.println(EntityUtils.toString(entity));
                
                EntityUtils.consume(entity);

                System.out.println("Initial set of cookies:");
                List<Cookie> cookies = cookieStore.getCookies();
                if (cookies.isEmpty()) {
                    System.out.println("None");
                } else {
                    for (int i = 0; i < cookies.size(); i++) {
                        System.out.println("- " + cookies.get(i).toString());
                    }
                }
                
                
            } finally {
                response1.close();
            }

            HttpPost httpost = new HttpPost("http://acm.hdu.edu.cn/vip/userloginex.php?action=login&cid=479");
            //HttpPost httpost = new HttpPost("http://www.rfp.ca/login/");
            
            List <NameValuePair> nvps = new ArrayList <NameValuePair>();
            nvps.add(new BasicNameValuePair("username", "admin"));
            nvps.add(new BasicNameValuePair("userpass", "921@njust"));
            
            //nvps.add(new BasicNameValuePair("return_url", ""));
            //nvps.add(new BasicNameValuePair("username", "774367334@qq.com"));
            //nvps.add(new BasicNameValuePair("password", "wjq2718281828"));
            //nvps.add(new BasicNameValuePair("action", "login"));

            httpost.setEntity(new UrlEncodedFormEntity(nvps, Consts.UTF_8));

            CloseableHttpResponse response2 = httpclient.execute(httpost);
            try {
                HttpEntity entity = response2.getEntity();

                System.out.println("Login form get: " + response2.getStatusLine());
                
                if (entity != null) {

                    InputStream is = entity.getContent();
                    BufferedReader br = new BufferedReader(new InputStreamReader(is));
                    String str ="";
                    while ((str = br.readLine()) != null){
                        System.out.println(""+str);
                    }
                }
                System.out.println("-------------------------------");
                Header[] hh = httpost.getAllHeaders();
                for(int i=0;i<hh.length;i++) {
                	System.out.println(hh[i]);
                }
                System.out.println("-------------------------------");
                Header header[] = response2.getAllHeaders();
                for(int i=0;i<header.length;i++) {
                	System.out.println(header[i]);
                }
                System.out.println("-------------------------------");
                
                EntityUtils.consume(entity);

                System.out.println("Post logon cookies:");
                List<Cookie> cookies = cookieStore.getCookies();
                if (cookies.isEmpty()) {
                    System.out.println("None");
                } else {
                    for (int i = 0; i < cookies.size(); i++) {
                        System.out.println("- " + cookies.get(i).toString());
                    }
                }
                
            } finally {
                response2.close();
            }
           
        } finally {
            //httpclient.close();
        }
        

    }
	
	@SuppressWarnings("deprecation")
	public static void test2() throws Exception{
		DefaultHttpClient httpclient = new DefaultHttpClient();
		httpclient.setRedirectStrategy(new DefaultRedirectStrategy() {                
	        public boolean isRedirected(HttpRequest request, HttpResponse response, HttpContext context)  {
	            boolean isRedirect=false;
	            try {
	                isRedirect = super.isRedirected(request, response, context);
	            } catch (ProtocolException e) {
	                // TODO Auto-generated catch block
	                e.printStackTrace();
	            }
	            if (!isRedirect) {
	                int responseCode = response.getStatusLine().getStatusCode();
	                if (responseCode == 301 || responseCode == 302) {
	                    return true;
	                }
	            }
	            return isRedirect;
	        }
	    });
		
	    CookieStore cookieStore = new BasicCookieStore();
	    httpclient.getParams().setParameter(
	      ClientPNames.COOKIE_POLICY, CookiePolicy.BROWSER_COMPATIBILITY); 
	    HttpContext context = new BasicHttpContext();
	    context.setAttribute(ClientContext.COOKIE_STORE, cookieStore);
	    //ResponseHandler<String> responseHandler = new BasicResponseHandler();

	    Credentials testsystemCreds = new UsernamePasswordCredentials("admin",  "921@njust");
	    httpclient.getCredentialsProvider().setCredentials(
	            new AuthScope(AuthScope.ANY_HOST, AuthScope.ANY_PORT),
	            testsystemCreds);

	    HttpPost postRequest = new HttpPost("http://acm.hdu.edu.cn/vip/userloginex.php?action=login&cid=479");
	    List<NameValuePair> formparams = new ArrayList<NameValuePair>();
	    formparams.add(new BasicNameValuePair("username", "admin"));
	    formparams.add(new BasicNameValuePair("userpass", "921@njust"));
	    postRequest.setEntity(new UrlEncodedFormEntity(formparams, "UTF-8"));
	    HttpResponse response = httpclient.execute(postRequest, context);
	    System.out.println(response);
	    
	    

	    if (response.getStatusLine().getStatusCode() != HttpStatus.SC_OK)
	        throw new IOException(response.getStatusLine().toString());

	    HttpUriRequest currentReq = (HttpUriRequest) context.getAttribute( 
	            ExecutionContext.HTTP_REQUEST);
	    HttpHost currentHost = (HttpHost)  context.getAttribute( 
	            ExecutionContext.HTTP_TARGET_HOST);
	    String currentUrl = currentHost.toURI() + currentReq.getURI();        
	    System.out.println(currentUrl);

	    HttpEntity entity = response.getEntity();
	    if (entity != null) {
	        long len = entity.getContentLength();
	        if (len != -1 && len < 2048) {
	            System.out.println(EntityUtils.toString(entity));
	        } else {
	            // Stream content out
	        }
	    }
	    if (entity != null) {

            InputStream is = entity.getContent();
            BufferedReader br = new BufferedReader(new InputStreamReader(is));
            String str ="";
            while ((str = br.readLine()) != null){
                System.out.println(""+str);
            }
        }
	}
	
	public static void storeToFile(String content, String fileName) throws Exception{
		FileWriter fw = new FileWriter(new File(fileName));
		fw.append(content);
		fw.close();
	}
	
	public static void main(String[] args) throws Exception{
		
		
		

	}
}
