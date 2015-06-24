package edu.zju.dd.crawler;

import java.io.File;
import java.io.PrintStream;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.util.EntityUtils;

import edu.zju.dd.codeforce.db.StatusSql;
import edu.zju.dd.codeforce.db.VisitedPageSql;
import edu.zju.dd.codeforce.meta.Pair;
import edu.zju.dd.codeforce.meta.Status;
import edu.zju.dd.codeforce.statusparser.StatusParser;

public class CodeForceStatusCrawler {

	private static final String LOG_FILE = "e:/status_log.out";

	public static String getHtml(String url) throws Exception {
		HttpClient httpClient = new DefaultHttpClient();
		HttpGet httpGet = new HttpGet(url);
		HttpResponse response = httpClient.execute(httpGet);
		HttpEntity entity = response.getEntity();
		return EntityUtils.toString(entity);
	}

	private static BlockingQueue<Pair<Integer, String>> bQueue = new LinkedBlockingQueue<Pair<Integer, String>>();

	class HtmlGetter extends Thread {
		public static final int ThreadNumber = 3;
		public static final int BasePageId = 1;

		private int threadId;
		public boolean stop;

		public HtmlGetter(int threadId) {
			this.threadId = threadId;
		}

		@Override
		public void run() {
			System.err.println("threadId = " + threadId + " is running...");
			stop = false;
			int sleep_time = 20 + 3000*threadId; //20ms + penalty
			for (int loop = 0; !stop; loop++) {
				try {
					int pageId = BasePageId + ThreadNumber * loop + threadId;
					if (VisitedPageSql.isVisited(pageId)) {
						continue;
					}
					System.err.println("thread " + threadId + " sleep for " + sleep_time
							+ " ms");
					sleep(sleep_time);
					//oo.wait(sleep_time);
					sleep_time *= 4;
					sleep_time = Math.min(sleep_time, 7200000); //������Сʱ����һ�� 
					System.err.println("thread " + threadId
							+ " is downloading status page " + pageId);
					String url = "http://codeforces.com/problemset/status/page/" + pageId
							+ "?order=BY_ARRIVED_ASC";
					long t = System.currentTimeMillis();
					bQueue.add(new Pair<Integer, String>(pageId, getHtml(url)));
					System.err.println(pageId);
					long tot = System.currentTimeMillis() - t;
					System.out.println("page = " + pageId + ", threadId = " + threadId
							+ ", time used = " + tot + " ms");
					sleep_time = 20;
				} catch (Exception e) {
					e.printStackTrace();
					loop--;
				}
			}
			System.err.println("threadId = " + threadId + " is shutdown.");
		}
	}

	private boolean stop_parser;

	private void runCrawler() throws Exception {
		PrintStream out = new PrintStream(new File(LOG_FILE));
		System.setOut(out);

		HtmlGetter htmlGetter[] = new HtmlGetter[HtmlGetter.ThreadNumber];
		for (int i = 0; i < HtmlGetter.ThreadNumber; i++) {
			htmlGetter[i] = new HtmlGetter(i);
			htmlGetter[i].start();
		}
		stop_parser = false;

		new Thread() {
			@Override
			public void run() {
				Scanner scanner = new Scanner(System.in);
				while (scanner.hasNext()) {
					String line = scanner.nextLine();
					if (line.equals("quit")) {
						stop_parser = true;
						break;
					} else {
						System.err.println("Instruction can't be parsed!");
					}
				}
				System.err.println("deamon thread stopped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}
		}.start();

		while (true) {
			while (bQueue.size() == 0 && !stop_parser) {
				Thread.currentThread().sleep(20);
			}
			if (stop_parser)
				break;
			Pair<Integer, String> pair = bQueue.remove();
			int pageId = pair.getFirst();
			String html = pair.getSecond();

			try {
				long t = System.currentTimeMillis();
				List<Status> list = StatusParser.parseStatusFromHtml(html);
				for (int j = 0; j < list.size(); j++) {
					StatusSql.insert(list.get(j));
				}
				if (list.size() == 50) {
					VisitedPageSql.insert(pageId); //����ٲ���
				} else {
					System.err.println("list size too small, size = "+list.size());
					break;
				}
				System.err.println("Parse and insert pageId = " + pageId + ", time used: "
						+ (System.currentTimeMillis() - t) + " ms");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		System.err.println("Main thread stoping...");
		for (int i = 0; i < HtmlGetter.ThreadNumber; i++) {
			htmlGetter[i].stop = true;
		}
		for (int i = 0; i < HtmlGetter.ThreadNumber; i++) {
			htmlGetter[i].join();
		}
	}

	public static void main(String[] args) throws Exception {
		CodeForceStatusCrawler instance = new CodeForceStatusCrawler();
		instance.runCrawler();
	}
}
